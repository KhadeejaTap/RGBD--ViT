
import torch
import torch.nn.functional as F
import math

from main.alignment import align_points_scale_z_shift, align_points_scale_xyz_shift


# ============================================================
# Loss functions implementing Kim et al. (CVPR 2026) Eq. 4-7
# L = L1 + Lg + Ll + Lm
# ============================================================

# Dataset-wide log-spaced depth bins for consistent subsampling
# Sensor range: 1mm - 30000mm
DEPTH_MIN = 1.0
DEPTH_MAX = 30000.0
NUM_DEPTH_BINS = 10
DEPTH_BIN_EDGES = torch.logspace(math.log10(DEPTH_MIN), math.log10(DEPTH_MAX), NUM_DEPTH_BINS + 1)


def _subsample_balanced(err: torch.Tensor, weight: torch.Tensor, depth: torch.Tensor,
                         valid_mask: torch.Tensor, max_samples: int = 10000,
                         bin_edges: torch.Tensor = None, near_bias: float = 0.5):
    """
    Balanced subsampling across log-spaced depth bins with optional near bias.

    Args:
        err: (N,) error values
        weight: (N,) weights
        depth: (N,) ground truth depth
        valid_mask: (N,) boolean valid mask
        max_samples: total budget
        bin_edges: (num_bins+1,) log-spaced bin edges
        near_bias: 0.0 = equal quota per bin, 1.0 = quota ∝ 1/depth_bin_center
                   0.5 = moderate near preference (near bins get ~2x far bins)

    Returns:
        mean error over balanced subsample
    """
    if bin_edges is None:
        bin_edges = DEPTH_BIN_EDGES.to(depth.device)

    num_bins = bin_edges.numel() - 1

    # Compute bin centers for weighting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # (num_bins,)

    # Quota per bin: base * (near_bias * (1/depth) + (1-near_bias) * 1)
    # Normalized so sum(quota) = max_samples
    inv_depth = 1.0 / bin_centers.clamp_min(1.0)
    inv_depth_norm = inv_depth / inv_depth.mean()
    weights = near_bias * inv_depth_norm + (1.0 - near_bias)
    weights = weights / weights.sum() * max_samples
    quota_per_bin = weights.long().clamp_min(1)  # at least 1 per bin

    # Bucketize depths into bins
    bin_idx = torch.bucketize(depth, bin_edges, right=True) - 1  # 0 to num_bins-1
    bin_idx = bin_idx.clamp(0, num_bins - 1)

    chosen = []
    for b in range(num_bins):
        bin_mask = (bin_idx == b) & valid_mask
        bin_indices = torch.where(bin_mask)[0]
        if bin_indices.numel() == 0:
            continue
        n_take = min(quota_per_bin[b].item(), bin_indices.numel())
        # Random permutation for unbiased sampling within bin
        perm = torch.randperm(bin_indices.numel(), device=depth.device)[:n_take]
        chosen.append(bin_indices[perm])

    if not chosen:
        return err[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=err.device)

    chosen = torch.cat(chosen)
    return (err[chosen] * weight[chosen]).sum() / weight[chosen].sum().clamp_min(1e-6)


def depth_weighted_l1_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6, max_samples: int = 10000):
    """
    Eq. 5: Depth-weighted L1 loss on raw metric depth maps.
    L1 = sum_{i in M} (1/d_i) * |d̃_i - d_i|

    Uses balanced subsampling across log-spaced depth bins for efficiency on dense maps.

    Args:
        pred_depth: (B, 1, H, W) predicted metric depth
        gt_depth: (B, 1, H, W) ground truth metric depth
        gt_mask: (B, 1, H, W) ground truth validity mask (1 for valid)
        eps: clamp minimum for inverse depth weight
        max_samples: max points per batch element for loss computation

    Returns:
        loss: scalar tensor
        misc: dict with loss value for logging
    """
    B, _, H, W = pred_depth.shape
    device = pred_depth.device

    # Valid mask: use provided gt_mask (1 for valid)
    valid_mask = gt_mask.bool()  # (B, 1, H, W)

    # For safe computation, replace invalid depths with 1.0 to avoid NaN/inf
    # This doesn't affect the final loss because invalid positions are masked out later
    # Actually doesnt do anythign bc we alr filled nans with 12mm
    gt_depth_safe = gt_depth.clone()

    # Inverse depth weighting: 1/d_i (using safe depths)
    weight = 1.0 / gt_depth_safe.clamp_min(eps)  # (B, 1, H, W)

    # Weighted absolute error (using safe depth to avoid NaN)
    err = weight * (pred_depth - gt_depth_safe).abs()  # (B, 1, H, W)
    err = err * valid_mask  # Zero out invalid positions

    # Flatten for subsampling
    err_flat = err.reshape(B, -1)           # (B, N)
    weight_flat = weight.reshape(B, -1)     # (B, N)
    valid_flat = valid_mask.reshape(B, -1)  # (B, N)
    gt_depth_flat = gt_depth.reshape(B, -1) # (B, N)  # Keep original for subsampling

    bin_edges = DEPTH_BIN_EDGES.to(device)

    total_loss = torch.tensor(0.0, device=device)
    for b in range(B):
        batch_loss = _subsample_balanced(
            err_flat[b], weight_flat[b], gt_depth_flat[b], valid_flat[b],
            max_samples, bin_edges, near_bias=0.5
        )
        total_loss += batch_loss

    loss = total_loss / B
    return loss, {"l1": loss.item()}


def global_scale_invariant_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, gt_mask: torch.Tensor,
                                 fx: float, fy: float, cx: float, cy: float,
                                 eps: float = 1e-6, max_samples: int = 10000):
    """
    Eq. 6 (global term): Global scale-invariant loss with scale + z-shift alignment.
    Lg = sum_{i in M} (1/d_i) * ||s * p̃_i - p_i||_1

    Uses closed-form weighted least squares for scale + z-shift (O(N) memory),
    then balanced subsampling across log-spaced depth bins for the L1 error term.

    Args:
        pred_depth: (B, 1, H, W) predicted metric depth
        gt_depth: (B, 1, H, W) ground truth metric depth
        gt_mask: (B, 1, H, W) ground truth validity mask (1 for valid)
        fx, fy, cx, cy: camera intrinsics
        eps: clamp minimum
        max_samples: max points per batch element for loss computation

    Returns:
        loss: scalar tensor
        misc: dict with loss value for logging
    """
    B, _, H, W = pred_depth.shape
    device = pred_depth.device

    # Backproject to 3D points
    pred_points = backproject(pred_depth, fx, fy, cx, cy)  # (B, 3, H, W)
    gt_points = backproject(gt_depth, fx, fy, cx, cy)      # (B, 3, H, W)

    # Valid mask: use provided gt_mask
    valid_mask = gt_mask.bool()  # (B, 1, H, W)
    valid_mask_3d = valid_mask.expand_as(pred_points)      # (B, 3, H, W)

    # For safe backprojection, replace invalid depths with 1.0 to avoid NaN/inf in 3D points
    # This doesn't affect the final loss because invalid positions are masked out later
    gt_depth_safe = gt_depth.clone()
    gt_points_safe = backproject(gt_depth_safe, fx, fy, cx, cy)  # (B, 3, H, W)

    # Reshape to (B, N, 3)
    pred_points_flat = pred_points.permute(0, 2, 3, 1).reshape(B, -1, 3)  # (B, H*W, 3)
    gt_points_flat = gt_points_safe.permute(0, 2, 3, 1).reshape(B, -1, 3)  # (B, H*W, 3)  # Use safe points
    valid_flat = valid_mask_3d.permute(0, 2, 3, 1).reshape(B, -1, 3)      # (B, H*W, 3)
    gt_depth_flat = gt_depth.permute(0, 2, 3, 1).reshape(B, -1)           # (B, H*W)  # Keep original for depth values

    # Per-pixel inverse depth weight (using safe depths for weight computation)
    gt_depth_safe_flat = gt_depth_safe.permute(0, 2, 3, 1).reshape(B, -1)  # (B, H*W)
    weight = (1.0 / gt_depth_safe_flat.clamp_min(eps)) * valid_flat[..., 0].float()  # (B, N)

    bin_edges = DEPTH_BIN_EDGES.to(device)

    total_loss = torch.tensor(0.0, device=device)
    valid_batches = 0
    for b in range(B):
        # Valid indices for this batch element
        valid_idx = torch.where(valid_flat[b, :, 0])[0]
        valid_count = valid_idx.numel()
        if valid_count == 0:
            continue

        pred_xyz = pred_points_flat[b, valid_idx]  # (N_valid, 3)
        gt_xyz = gt_points_flat[b, valid_idx]      # (N_valid, 3)
        w = weight[b, valid_idx]                   # (N_valid,)
        d = gt_depth_flat[b, valid_idx]            # (N_valid,)  # Use original depth for subsampling

        # Closed-form weighted least squares for scale + z-shift
        # Scale: sum(w * pred · gt) / sum(w * pred · pred)  -- dot product over XYZ
        w_unsq = w.unsqueeze(-1)  # (N, 1)
        num = (w_unsq * pred_xyz * gt_xyz).sum(dim=-1).sum()  # scalar
        den = (w_unsq * pred_xyz * pred_xyz).sum(dim=-1).sum().clamp_min(eps)  # scalar
        scale = num / den

        # Z-shift: weighted mean of (gt_z - scale * pred_z)
        pred_z = pred_xyz[:, 2]
        gt_z = gt_xyz[:, 2]
        shift_z = ((gt_z - scale * pred_z) * w).sum() / w.sum().clamp_min(eps)

        # Apply alignment: s * pred + [0, 0, shift_z]
        aligned_pred = scale * pred_xyz
        aligned_pred[:, 2] += shift_z

        # L1 error over xyz, weighted by inverse depth
        err = (aligned_pred - gt_xyz).abs().sum(dim=-1) * w  # (N_valid,)

        # Balanced subsampling across log-spaced depth bins
        batch_loss = _subsample_balanced(err, w, d, torch.ones_like(d, dtype=torch.bool),
                                         max_samples, bin_edges, near_bias=0.5)
        total_loss += batch_loss
        valid_batches += 1

    loss = total_loss / max(valid_batches, 1)
    return loss, {"lg": loss.item()}


def local_scale_invariant_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, gt_mask: torch.Tensor,
                                fx: float, fy: float, cx: float, cy: float,
                                focal: float = None,  # kept for signature compatibility
                                level: int = 2,
                                radius_2d: float = 0.05,
                                radius_3d: float = 0.1,
                                min_points_per_patch: int = 16,
                                eps: float = 1e-6,
                                local_loss_downsample: int = 2): # lowered frm 4 to 2, bc res is low rn
    """
    Eq. 6 (local term): Local scale-invariant loss with per-patch scale + xyz-shift.
    Ll = sum_{j in H} sum_{i in S_j} (1/d_i) * ||s_j * p̃_i - p_i||_1

    Uses align_points_scale_xyz_shift for per-patch scale + xyz-shift alignment.
    Keeps existing patch anchor-sampling logic from codebase.

    Args:
        pred_depth: (B, 1, H, W) predicted metric depth
        gt_depth: (B, 1, H, W) ground truth metric depth
        gt_mask: (B, 1, H, W) ground truth validity mask (1 for valid)
        fx, fy, cx, cy: camera intrinsics
        focal: focal length (for compatibility, unused)
        level: pyramid level for anchor sampling
        radius_2d: 2D radius for patch sampling
        radius_3d: 3D radius for patch sampling
        min_points_per_patch: minimum points per patch
        eps: clamp minimum
        local_loss_downsample: downsample factor for local loss (default 4).
                               Runs local loss at H/downsample x W/downsample.

    Returns:
        loss: scalar tensor
        misc: dict with loss value for logging
    """
    B, _, H, W = pred_depth.shape
    device = pred_depth.device

    # Downsample for local loss to avoid OOM at high resolution
    if local_loss_downsample > 1:
        ds = local_loss_downsample
        pred_depth = F.interpolate(pred_depth, size=(H // ds, W // ds), mode="nearest")
        gt_depth = F.interpolate(gt_depth, size=(H // ds, W // ds), mode="nearest")
        gt_mask = F.interpolate(gt_mask, size=(H // ds, W // ds), mode="nearest")
        fx, fy, cx, cy = fx / ds, fy / ds, cx / ds, cy / ds
        H, W = H // ds, W // ds

    # Backproject to 3D points
    pred_points = backproject(pred_depth, fx, fy, cx, cy)  # (B, 3, H, W)

    # For safe backprojection, replace invalid depths with 1.0 to avoid NaN/inf in 3D points
    # This doesn't affect the final loss because invalid positions are masked out later
    gt_depth_safe = gt_depth.clone()
    gt_depth_safe = torch.nan_to_num(gt_depth_safe, nan=1.0, posinf=1.0, neginf=1.0)
    gt_depth_safe[~gt_mask.bool()] = 1.0  # Safe value for depth (avoids division by zero or NaN)
    gt_points = backproject(gt_depth_safe, fx, fy, cx, cy)      # (B, 3, H, W)

    # Valid mask: use provided gt_mask
    valid_mask = gt_mask.bool()  # (B, 1, H, W)

    # Generate anchor points using existing sampling logic
    # Sample anchors at pyramid level
    stride = 2 ** level
    anchor_h = H // stride
    anchor_w = W // stride

    # Create grid of anchor points
    ys = torch.arange(stride // 2, H, stride, device=device, dtype=torch.long)
    xs = torch.arange(stride // 2, W, stride, device=device, dtype=torch.long)

    if len(ys) == 0 or len(xs) == 0:
        # Fallback: single patch covering whole image
        ys = torch.tensor([H // 2], device=device)
        xs = torch.tensor([W // 2], device=device)

    total_loss = torch.tensor(0.0, device=device)
    total_patches = torch.tensor(0, device=device)

    for y in ys:
        for x in xs:
            # Define patch boundaries in 2D
            y_min = max(0, y - int(radius_2d * H))
            y_max = min(H, y + int(radius_2d * H) + 1)
            x_min = max(0, x - int(radius_2d * W))
            x_max = min(W, x + int(radius_2d * W) + 1)

            # Extract patch
            pred_patch = pred_points[:, :, y_min:y_max, x_min:x_max]  # (B, 3, ph, pw)
            gt_patch = gt_points[:, :, y_min:y_max, x_min:x_max]
            mask_patch = valid_mask[:, :, y_min:y_max, x_min:x_max]
            depth_patch = gt_depth[:, :, y_min:y_max, x_min:x_max]  # Use original depth for values

            # Flatten patch
            ph, pw = pred_patch.shape[-2:]
            pred_flat = pred_patch.permute(0, 2, 3, 1).reshape(B, -1, 3)  # (B, N_patch, 3)
            gt_flat = gt_patch.permute(0, 2, 3, 1).reshape(B, -1, 3)
            mask_flat = mask_patch.permute(0, 2, 3, 1).reshape(B, -1)
            depth_flat = depth_patch.permute(0, 2, 3, 1).reshape(B, -1)

            # Weight: inverse depth * valid mask (using safe depths for weight computation)
            depth_patch_safe = depth_patch.clone()
            depth_patch_safe = torch.nan_to_num(depth_patch_safe, nan=1.0, posinf=1.0, neginf=1.0)
            depth_patch_safe[~mask_patch.bool()] = 1.0  # Safe value for depth in patch
            depth_patch_safe_flat = depth_patch_safe.permute(0, 2, 3, 1).reshape(B, -1)
            weight = (1.0 / depth_patch_safe_flat.clamp_min(eps)) * mask_flat.float()  # (B, N_patch)

            # Skip if too few valid points
            valid_count = mask_flat.sum(dim=-1)
            if (valid_count < min_points_per_patch).all():
                continue

            # Align with scale + xyz-shift per patch
            scale, shift = align_points_scale_xyz_shift(pred_flat, gt_flat, weight)
            # scale: (B,), shift: (B, 3)

            # Apply alignment
            aligned_pred = scale.view(B, 1, 1) * pred_flat + shift.view(B, 1, 3)

            # L1 error over xyz, weighted
            err = (aligned_pred - gt_flat).abs().sum(dim=-1)  # (B, N_patch)
            err = err * weight

            # Mean over valid points in patch
            patch_valid = weight.sum(dim=-1).clamp_min(1.0)
            patch_loss = err.sum(dim=-1) / patch_valid  # (B,)

            # Only count patches with enough points
            patch_mask = (valid_count >= min_points_per_patch).float()
            total_loss = total_loss + (patch_loss * patch_mask).sum()
            total_patches = total_patches + patch_mask.sum()

    if total_patches == 0:
        # DEBUG: Print info about why no patches were valid
        print(f"DEBUG local loss: H={H}, W={W}, level={level}, stride={stride}")
        print(f"DEBUG local loss: len(ys)={len(ys)}, len(xs)={len(xs)}")
        print(f"DEBUG local loss: min_points_per_patch={min_points_per_patch}")
        if len(ys) > 0 and len(xs) > 0:
            # Check first patch
            y_first = ys[0].item() if len(ys) > 0 else 0
            x_first = xs[0].item() if len(xs) > 0 else 0
            y_min = max(0, y_first - int(radius_2d * H))
            y_max = min(H, y_first + int(radius_2d * H) + 1)
            x_min = max(0, x_first - int(radius_2d * W))
            x_max = min(W, x_first + int(radius_2d * W) + 1)
            print(f"DEBUG local loss: first patch y[{y_min}:{y_max}], x[{x_min}:{x_max}]")
            if 'mask_flat' in locals():
                print(f"DEBUG local loss: mask_flat shape={mask_flat.shape}")
                print(f"DEBUG local loss: valid_count for first patch={mask_flat[0].sum().item()}")
        return torch.tensor(0.0, device=device, requires_grad=True), {"ll": 0.0}

    loss = total_loss / torch.clamp(total_patches, min=1.0)
    return loss, {"ll": loss.item()}


def mask_bce_loss(mask_logit: torch.Tensor, mask_gt: torch.Tensor):
    """
    Eq. 7: Binary cross-entropy for validity mask.
    Lm = -sum_i [m_i * log(m̃_i) + (1-m_i) * log(1-m̃_i)]
    """
    lm_loss = F.binary_cross_entropy_with_logits(mask_logit, mask_gt)
    return lm_loss, {"lm": lm_loss.item()}


def total_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor,
               gt_mask: torch.Tensor, mask_logit: torch.Tensor, mask_gt: torch.Tensor,
               fx: float, fy: float, cx: float, cy: float,
               focal: float = None, level: int = 2, radius_2d: float = 0.05,
               radius_3d: float = 0.1, min_points_per_patch: int = 16,
               eps: float = 1e-6, local_loss_downsample: int = 4):
    """
    Eq. 4: L = L1 + Lg + Ll + Lm (equal weighting)

    Args:
        pred_depth: (B, 1, H, W) predicted metric depth
        gt_depth: (B, 1, H, W) ground truth metric depth
        gt_mask: (B, 1, H, W) ground truth validity mask (1 for valid)
        mask_logit: (B, 1, H, W) predicted mask logits
        mask_gt: (B, 1, H, W) ground truth mask
        fx, fy, cx, cy: camera intrinsics
        focal, level, radius_2d, radius_3d, min_points_per_patch: local loss params
        eps: numerical stability
        local_loss_downsample: downsample factor for local loss (default 4)

    Returns:
        total_loss: scalar tensor
        misc: dict with individual loss components
    """
    l1, misc1 = depth_weighted_l1_loss(pred_depth, gt_depth, gt_mask, eps)
    lg, misc2 = global_scale_invariant_loss(pred_depth, gt_depth, gt_mask, fx, fy, cx, cy, eps)
    ll, misc3 = local_scale_invariant_loss(
        pred_depth, gt_depth, gt_mask, fx, fy, cx, cy,
        focal=focal, level=level, radius_2d=radius_2d,
        radius_3d=radius_3d, min_points_per_patch=min_points_per_patch, eps=eps,
        local_loss_downsample=local_loss_downsample
    )
    lm, misc4 = mask_bce_loss(mask_logit, mask_gt)

    total = l1 + lg + ll + lm
    misc = {**misc1, **misc2, **misc3, **misc4}
    return total, misc


def backproject(depth: torch.Tensor, fx: float, fy: float, cx: float, cy: float):
    """
    Backproject depth to 3D points using pinhole camera model.
    depth: (B, 1, H, W) -> points: (B, 3, H, W)
    """
    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    u = u.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    v = v.unsqueeze(0).unsqueeze(0)

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return torch.cat([X, Y, Z], dim=1)  # (B, 3, H, W)


if __name__ == "__main__":
    # Quick test
    B, H, W = 2, 64, 64
    fx, fy, cx, cy = 100.0, 100.0, 32.0, 32.0

    pred_depth = torch.rand(B, 1, H, W) * 5000 + 300
    gt_depth = torch.rand(B, 1, H, W) * 5000 + 300
    valid_mask = (torch.rand(B, 1, H, W) > 0.2).float()
    mask_logit = torch.randn(B, 1, H, W)
    mask_gt = (torch.rand(B, 1, H, W) > 0.2).float()

    loss, parts = total_loss(pred_depth, gt_depth, valid_mask, mask_logit, mask_gt,
                             fx, fy, cx, cy)
    print("Total loss:", loss.item())
    print("Parts:", parts)

    assert not torch.isnan(loss), "NaN in total loss"
    print("PASSED: no NaN")
