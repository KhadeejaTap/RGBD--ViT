from torch.utils.data import Dataset
Dataset.__class_getitem__ = classmethod(lambda cls, *args: cls)
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from main.encoder import load_backbone, crop_to_multiple, crop_to_original
from main.fusionv3 import DualBranchEncoder
from main.decoder import DPTDecoder, denormalize_depth
from main.preprocessed_dataset import PreprocessedHypersimDataset
from main.normalize import compute_log_params
from main.intrinsics import get_intrinsics
from main.losses import total_loss


def save_mask_png(mask_prob: torch.Tensor, path: Path):
    """Save validity mask probability as 8-bit PNG.

    Args:
        mask_prob: Torch tensor with values in [0,1] representing validity probability
        path: Path object specifying where to save the PNG file
    """
    mask = mask_prob.detach().cpu().numpy()
    mask = np.squeeze(mask)
    mask = np.clip(mask, 0.0, 1.0)  # Ensure valid probability range
    mask_uint8 = (mask * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_uint8, mode="L").save(path)  # Grayscale 8-bit

NUM_ITERS = 1000
ENCODER_LR = 1e-5   # rolled back from 1e-3 -- that value diverged (see run log)
DECODER_LR = 1e-4   # rolled back from 1e-2 -- that value diverged (see run log)
GRAD_CLIP_MAX_NORM = 1.0

# save a checkpoint prediction at these iterations so sharpness progression
# can be inspected in one run instead of re-running from scratch each time
CHECKPOINT_ITERS = [1, 50, 150, 300]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("--- device ---")
    print(DEVICE)

    print("\n--- loading sample ---")
    data_dir = str(Path(__file__).resolve().parent.parent / "hypersim_test_data")
    ds = PreprocessedHypersimDataset(data_dir=data_dir)
    if len(ds) == 0:
        raise FileNotFoundError("No NPZ files found in " + data_dir)
    sample = ds.__getitem__(0)
    orig_h, orig_w = sample["rgb"].shape[-2], sample["rgb"].shape[-1]
    print("original resolution:", orig_h, orig_w)

    rgb = crop_to_multiple(sample["rgb"].unsqueeze(0)).to(DEVICE)
    gt_depth = crop_to_multiple(sample["gt_depth"].unsqueeze(0)).to(DEVICE)
    gt_mask = crop_to_multiple(sample["gt_mask"].unsqueeze(0)).to(DEVICE)
    out_h, out_w = rgb.shape[-2], rgb.shape[-1]
    h_patch, w_patch = out_h // 16, out_w // 16
    print("cropped resolution:", out_h, out_w, " patch grid:", h_patch, w_patch)

    fx, fy, cx, cy = get_intrinsics(out_w, out_h)
    print(f"intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    alpha, beta = compute_log_params()
    print(f"alpha={alpha:.4f} beta={beta:.4f}")
    depth_mm = sample["depth_filled_mm"].clamp_min(1.0)  # (1, H, W)
    zhat = ((torch.log(depth_mm) - beta) / alpha).clamp(0.0, 1.0)  # (1, H, W)
    valid_mask = sample["valid_mask"].float()  # (H, W)

    depth_input_raw = torch.stack([zhat.squeeze(0), zhat.squeeze(0), valid_mask], dim=0)  # (3, H, W)
    depth_input = crop_to_multiple(depth_input_raw.unsqueeze(0) * 2.0 - 1.0).to(DEVICE)  # (1, 3, H, W)

    print("\n--- building model ---")
    model_rgb = load_backbone()
    model_depth = load_backbone()
    encoder = DualBranchEncoder(model_rgb, model_depth).to(DEVICE)
    decoder = DPTDecoder().to(DEVICE)

    all_params = list(encoder.parameters()) + list(decoder.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": ENCODER_LR},
        {"params": decoder.parameters(), "lr": DECODER_LR},
    ])

    print(f"encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    print("\n--- overfit loop ---")
    checkpoints = {}  # iter -> (depth_pred_mm, mask_pred)

    for it in range(1, NUM_ITERS + 1):
        optimizer.zero_grad()

        features = encoder(rgb, depth_input)
        depth_hat, mask_logit = decoder(features, h_patch, w_patch, out_h, out_w)

        # inputs and GT are already cropped to multiple of 16; use directly for loss
        depth_hat_c = depth_hat
        mask_logit_c = mask_logit

        depth_pred_metric = denormalize_depth(depth_hat_c, alpha, beta)

        loss, parts = total_loss(depth_pred_metric, gt_depth, gt_mask.float(), mask_logit_c, gt_mask.float(), fx, fy, cx, cy)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        if it == 1 or it % 10 == 0 or it == NUM_ITERS:
            with torch.no_grad():
                valid = gt_mask.bool()
                if valid.any():
                    abs_err = (depth_pred_metric - gt_depth).abs()[valid].mean().item()
                else:
                    abs_err = float("nan")
            print(
                f"iter {it:4d}  total={loss.item():.4f}  "
                f"l1={parts['l1']:.4f}  lg={parts['lg']:.4f}  "
                f"ll={parts['ll']:.4f}  lm={parts['lm']:.4f}  "
                f"mean_abs_depth_err_mm={abs_err:.2f}"
            )

        assert not torch.isnan(loss), f"NaN loss at iter {it}"

        if it in CHECKPOINT_ITERS:
            with torch.no_grad():
                checkpoints[it] = (
                    depth_pred_metric.detach().cpu().numpy().copy(),
                    torch.sigmoid(mask_logit_c).detach().cpu().numpy().copy(),
                )
            print(f"  [checkpoint saved at iter {it}]")

    print("\ndone. if working, total loss and mean_abs_depth_err_mm should trend down,")
    print("and predictions across checkpoints should visibly sharpen over iterations.")

    for it, (depth_pred, mask_pred) in checkpoints.items():
        np.save(f"prediction_depth_mm_iter{it}.npy", depth_pred)
        np.save(f"prediction_mask_iter{it}.npy", mask_pred)
        # Also save as PNG for visualization
        save_mask_png(torch.from_numpy(mask_pred), Path(f"prediction_mask_iter{it}.png"))
    print("saved checkpoints:", sorted(checkpoints.keys()))
    print("files: prediction_depth_mm_iter{N}.npy, prediction_mask_iter{N}.npy for N in", CHECKPOINT_ITERS)
    print("and prediction_mask_iter{N}.png for N in", CHECKPOINT_ITERS)


if __name__ == "__main__":
    main()
