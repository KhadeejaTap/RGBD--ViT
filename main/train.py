from torch.utils.data import Dataset
Dataset.__class_getitem__ = classmethod(lambda cls, *args: cls)
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import csv
import math
import os
import csv
import os
import time
from torch.utils.tensorboard import SummaryWriter


from main.encoder import load_backbone, crop_to_multiple, crop_to_original
from main.fusionv3 import DualBranchEncoder
from main.decoder import DPTDecoder, denormalize_depth
from main.preprocessed_dataset import PreprocessedHypersimDataset
from main.normalize import compute_log_params
from main.intrinsics import get_intrinsics
from main.losses import total_loss

data_dir = str(Path(__file__).resolve().parent.parent.parent / "preprocessed_hypersim_data")
out_dir = str(Path(__file__).resolve().parent.parent / "outputs")
# Set up device for single or multi-GPU training
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        DEVICE = torch.device("cuda:0")  # DataParallel will handle distribution
    else:
        DEVICE = torch.device("cuda:0")
        print("Using single GPU for training.")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU for training.")


def parse_args():
    p = argparse.ArgumentParser(description=" depth completion training")
    # Data
    p.add_argument("--data-dir", type=str, default=data_dir, help="Preprocessed data directory (if set, uses fast preprocessed dataset)")
    p.add_argument("--val-fraction", type=float, default=0.05, help="Validation scene fraction (default: 0.05)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--init", type=str, default="pretrained", choices=["pretrained", "random", "shared_random"],
                   help="Weight initialization (default: pretrained)")
    # Training
    p.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    p.add_argument("--enc_lr", type=float, default=1e-5, help="Encoder LR (default: 1e-5)")
    p.add_argument("--dec_lr", type=float, default=1e-4, help="Decoder LR (default: 1e-4)")
    p.add_argument("--grad-clip", type=float, default=0.1, help="Grad clip max norm (default: 0.1)")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay (default: 1e-2)")
    p.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights, train decoder only")
    p.add_argument("--lr-decay-steps", type=int, default=25000, help="Halve LR every N steps (default: 25000)")
    p.add_argument("--warmup-iters", type=int, default=1000, help="Linear LR warmup iterations (default: 1000)")
    # Loss
    p.add_argument("--no-lg", action="store_true", help="Disable global scale-invariant loss (lg)")
    p.add_argument("--local-downsample", type=int, default=2, help="Local loss downsample factor (default: 2)")
    p.add_argument("--level", type=int, default=2, help="Local loss pyramid level (default: 2)")
    p.add_argument("--radius-2d", type=float, default=0.05, help="Local loss 2D radius (default: 0.05)")
    p.add_argument("--min-points", type=int, default=4, help="Min points per patch (default: 4)")
    # Overfitting
    p.add_argument("--overfit", action="store_true", help="Overfit on a single sample (for debugging)")
    p.add_argument("--overfit-sample-idx", type=int, default=0, help="Index of sample to overfit on (default: 0)")
    p.add_argument("--overfit-iters", type=int, default=500, help="Number of iterations for overfitting (default: 500)")
    p.add_argument("--print-every", type=int, default=50, help="Print overfit metrics every N iterations (default: 50)")
    p.add_argument("--overfit-save-every", type=int, default=0,
                   help="Save overfit predictions every N iterations (0 = ten times per run)")
    # Checkpointing
    p.add_argument("--out-dir", type=str, default=out_dir, help="Output directory (default: outputs)")
    p.add_argument("--save-weights", action="store_true", default=True, help="Save model weights")
    p.add_argument("--no-save-weights", dest="save_weights", action="store_false", help="Disable weight saving")
    p.add_argument("--save-checkpoint-fraction", type=float, default=0.0, help="Save a checkpoint every N fractions of total iters (0 = disabled, 1.0 = after every iteration, e.g. 0.1 = every 10%% of total iters)")
    p.add_argument("--val-every-fraction", type=float, default=0.1, help="Run val pass every fraction of iters (default: 0.1)")
    p.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory to search for auto-resume checkpoints (default: checkpoints)")
    # Resume
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt to resume from")
    # Misc
    p.add_argument("--shuffle", action="store_true", help="Shuffle training data")  # Resume skipping is nondeterministic when enabled.
    p.add_argument("--num-workers", type=int, default=1, help="DataLoader workers (default: 1)")
    p.add_argument("--dry-run", action="store_true", help="Forward pass only, no training")
    p.add_argument("--timing", action="store_true", help="Print per-iteration timing breakdown")
    p.add_argument("--log-every", type=int, default=50, help="Log metrics to CSV every N iterations (default: 50)")
    # Testing
    p.add_argument("--test", action="store_true", help="Run inference on test data and save predictions")
    p.add_argument("--test-dir", type=str, default="hypersim_data_simple_nanfix", help="Directory containing test data (default: hypersim_data_simple_nanfix)")
    p.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint .pt file for testing (required if --test is used)")
    p.add_argument("--test-save-dir", type=str, default="test_predictions", help="Directory to save test predictions (default: test_predictions)")
    p.add_argument("--test-save-n", type=int, default=3, help="Number of test predictions to save as .npy files (default: 3)")
    # Epoch training
    p.add_argument("--epochs", type=int, default=0, help="Number of epochs to train for (0 = use iteration-based training, >0 = enable epoch-based training)")
    # TensorBoard logging
    p.add_argument("--use-tensorboard", action="store_true", help="Enable TensorBoard logging (disables CSV logging)")
    return p.parse_args()


def log_to_csv(csv_path, iteration, phase, loss, metrics=None, iter_time=None):
    """Append a line to the CSV log file."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if metrics is None:
            row = [iteration, phase, loss, '', '', '', '', '', '', iter_time if iter_time is not None else '']
        else:
            row = [
                iteration,
                phase,
                loss,
                metrics.get('absrel', ''),
                metrics.get('mae', ''),
                metrics.get('rmse', ''),
                metrics.get('delta1', ''),
                metrics.get('delta2', ''),
                metrics.get('delta3', ''),
                metrics.get('delta_1025', ''),
                iter_time if iter_time is not None else ''
            ]
        writer.writerow(row)


def build_model():
    """Build encoder and decoder."""
    model_rgb = load_backbone()
    model_depth = load_backbone()
    encoder = DualBranchEncoder(model_rgb, model_depth)
    decoder = DPTDecoder()

    # Wrap models with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)

    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    return encoder, decoder


def compute_metrics(depth_pred, gt_depth, gt_mask):
    """Compute raw-depth accuracy metrics from pred/gt depth.

    Includes the standard threshold accuracy metrics (delta1/delta2/delta3):
    the fraction of valid pixels where max(pred/gt, gt/pred) falls under
    1.25, 1.25^2, 1.25^3 respectively. delta1 is the usual headline number.

    Also includes delta_1025: the tighter threshold accuracy reported by some
    benchmarks, using the same max(pred/gt, gt/pred) ratio but a 1.025
    threshold instead of 1.25 -- i.e. rel = mean(|pred-gt|/gt) (== absrel
    here) and delta_1025 = fraction of pixels with ratio < 1.025.
    """
    valid = gt_mask.bool()
    if not valid.any():
        return {
            "absrel": float("nan"), "mae": float("nan"), "rmse": float("nan"),
            "delta1": float("nan"), "delta2": float("nan"), "delta3": float("nan"),
            "delta_1025": float("nan"),
        }

    pred_v = depth_pred[valid].clamp_min(1e-6)
    gt_v = gt_depth[valid].clamp_min(1e-6)
    diff = pred_v - gt_v
    abs_diff = diff.abs()

    absrel = (abs_diff / gt_v).mean().item()
    mae = abs_diff.mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()

    ratio = torch.max(pred_v / gt_v, gt_v / pred_v)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()
    delta_1025 = (ratio < 1.025).float().mean().item()

    return {
        "absrel": absrel, "mae": mae, "rmse": rmse,
        "delta1": delta1, "delta2": delta2, "delta3": delta3,
        "delta_1025": delta_1025,
    }


def compute_batch_metrics(depth_preds, prepared_batch):
    """Compute metrics across a batched prediction tensor."""
    metric_names = (
        "absrel", "mae", "rmse", "delta1", "delta2", "delta3", "delta_1025"
    )
    metric_sums = {key: 0.0 for key in metric_names}
    metric_counts = {key: 0 for key in metric_names}
    if depth_preds.dim() == 3:
        depth_preds = depth_preds.unsqueeze(0)
    if prepared_batch["gt_depth"].dim() == 3:
        gt_depth = prepared_batch["gt_depth"].unsqueeze(0)
        gt_mask = prepared_batch["gt_mask"].unsqueeze(0)
    else:
        gt_depth = prepared_batch["gt_depth"]
        gt_mask = prepared_batch["gt_mask"]

    for depth_pred, sample_gt_depth, sample_gt_mask in zip(
        depth_preds, gt_depth, gt_mask
    ):
        sample_metrics = compute_metrics(
            depth_pred, sample_gt_depth, sample_gt_mask
        )
        for key, value in sample_metrics.items():
            if value == value:
                metric_sums[key] += value
                metric_counts[key] += 1

    return {
        key: metric_sums[key] / metric_counts[key]
        if metric_counts[key] else float("nan")
        for key in metric_names
    }


def prepare_sample(samples, device):
    """Preprocess one same-resolution batch into device-resident tensors.

    Args:
        samples: Either a single sample dict or a list of sample dicts
        device: Target device for tensors

    Returns:
        A prepared batch dictionary with tensors whose first dimension is batch size.
    """
    if isinstance(samples, list):
        if not samples:
            raise ValueError("Cannot prepare an empty sample batch")
        batch = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            batch[key] = torch.stack(values) if all(torch.is_tensor(v) for v in values) else values
    else:
        batch = samples

    rgb = batch["rgb"].to(device)
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    gt_depth = batch["gt_depth"].to(device)
    if gt_depth.dim() == 3:
        gt_depth = gt_depth.unsqueeze(0)
    gt_depth = gt_depth / 1000.0
    gt_mask = batch["gt_mask"].to(device)
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(0)
    depth_filled_mm = batch["depth_filled_mm"].to(device)
    if depth_filled_mm.dim() == 3:
        depth_filled_mm = depth_filled_mm.unsqueeze(0)
    valid_mask = batch["valid_mask"].float().to(device)
    if valid_mask.dim() == 2:
        valid_mask = valid_mask.unsqueeze(0)

    orig_h, orig_w = rgb.shape[-2:]
    fx, fy, cx, cy = get_intrinsics(orig_w, orig_h)
    alpha, beta = compute_log_params(zmin=0.3, zmax=20.0)

    depth_m = depth_filled_mm / 1000.0
    depth_m = depth_m.clamp_min(0.001)
    zhat = ((torch.log(depth_m) - beta) / alpha).clamp(0.0, 1.0)
    depth_input = torch.cat([zhat, zhat, valid_mask.unsqueeze(1)], dim=1) * 2.0 - 1.0

    rgb_p = crop_to_multiple(rgb)
    depth_input_p = crop_to_multiple(depth_input)
    padded_h, padded_w = rgb_p.shape[-2:]
    return {
        "rgb_p": rgb_p,
        "depth_input_p": depth_input_p,
        "gt_depth": gt_depth,
        "gt_mask": gt_mask,
        "valid_mask": valid_mask,
        "alpha": alpha, "beta": beta,
        "orig_h": orig_h, "orig_w": orig_w,
        "padded_h": padded_h, "padded_w": padded_w,
        "h_patch": padded_h // 16, "w_patch": padded_w // 16,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
    }


def forward_and_loss(encoder, decoder, prepared_batch, args):
    """Run one batched forward pass and compute the batch loss.
    Returns (total_loss, avg_parts, depth_preds, mask_logits) where:
    - total_loss: sum of losses across batch (for backward)
    - avg_parts: average of parts dict across batch (for logging)
    - depth_preds: batched depth prediction tensor
    - mask_logits: batched mask-logit tensor
    """
    features = encoder(prepared_batch["rgb_p"], prepared_batch["depth_input_p"])
    depth_hat, mask_logit = decoder(
        features, prepared_batch["h_patch"], prepared_batch["w_patch"],
        prepared_batch["padded_h"], prepared_batch["padded_w"],
    )
    depth_hat = crop_to_original(
        depth_hat, prepared_batch["orig_h"], prepared_batch["orig_w"]
    )
    mask_logit = crop_to_original(
        mask_logit, prepared_batch["orig_h"], prepared_batch["orig_w"]
    )
    depth_pred = denormalize_depth(
        depth_hat, prepared_batch["alpha"], prepared_batch["beta"]
    )
    loss, parts = total_loss(
        depth_pred, prepared_batch["gt_depth"], prepared_batch["gt_mask"].float(),
        mask_logit, prepared_batch["valid_mask"].unsqueeze(1),
        prepared_batch["fx"], prepared_batch["fy"],
        prepared_batch["cx"], prepared_batch["cy"],
        level=args.level, radius_2d=args.radius_2d,
        min_points_per_patch=args.min_points,
        local_loss_downsample=args.local_downsample,
    )
    if torch.is_tensor(loss) and loss.dim() > 0:
        loss = loss.mean()
    if args.no_lg and torch.is_tensor(loss):
        loss = loss - torch.as_tensor(parts["lg"], device=loss.device)
        parts["lg"] = 0.0
    return loss, parts, depth_pred, mask_logit

#allows me to continue training
def save_checkpoint(path, *, train_step, global_iter, encoder, decoder, optimizer, scheduler, train_loss=None, val_loss=None, val_absrel=None, val_delta1=None, val_delta_1025=None):
    # Handle DataParallel models by accessing the underlying module
    encoder_state = encoder.module.state_dict() if isinstance(encoder, torch.nn.DataParallel) else encoder.state_dict()
    decoder_state = decoder.module.state_dict() if isinstance(decoder, torch.nn.DataParallel) else decoder.state_dict()

    payload = {
        "train_step": train_step,
        "global_iter": global_iter,
        "encoder_state_dict": encoder_state,
        "decoder_state_dict": decoder_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if train_loss is not None:
        payload["train_loss"] = train_loss
    if val_loss is not None:
        payload["val_loss"] = val_loss
    if val_absrel is not None:
        payload["val_absrel"] = val_absrel
    if val_delta1 is not None:
        payload["val_delta1"] = val_delta1
    if val_delta_1025 is not None:
        payload["val_delta_1025"] = val_delta_1025
    torch.save(payload, path)


def run_validation(encoder, decoder, val_ds, args, save_dir=None, save_n=3):
    """Run a full pass over val_ds and return averaged loss + metrics.

    If save_dir is given, saves depth_pred (and the sigmoid mask prob) as
    .npy for the first `save_n` val samples, for later qualitative analysis.
    """
    encoder.eval()
    decoder.eval()
    total_loss_sum = 0.0
    metric_sums = {"absrel": 0.0, "mae": 0.0, "rmse": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0, "delta_1025": 0.0}
    n = 0
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i in range(len(val_ds)):
            sample = val_ds[i]
            prepared = prepare_sample(sample, DEVICE)
            loss, parts, depth_pred, mask_logit = forward_and_loss(encoder, decoder, prepared, args)

            # Skip iteration if loss is NaN or infinite (robust check for tensor/non-tensor)
            is_invalid = False
            if torch.is_tensor(loss):
                if torch.isnan(loss) or torch.isinf(loss):
                    is_invalid = True
            else:
                # Handle non-tensor losses (numpy arrays, Python scalars, etc.)
                try:
                    loss_val = float(loss)
                    if math.isnan(loss_val) or math.isinf(loss_val):
                        is_invalid = True
                except (ValueError, TypeError):
                    # If we can't convert to float, treat as invalid
                    is_invalid = True

            if is_invalid:
                print(f"  WARNING: Invalid loss at val sample {i}, skipping (loss type: {type(loss)}, value: {loss})")
                continue

            metrics = compute_batch_metrics(depth_pred, prepared)

            # Safely convert loss to Python float for accumulation
            try:
                loss_value = loss.item() if torch.is_tensor(loss) else float(loss)
            except (ValueError, TypeError) as e:
                print(f"  WARNING: Could not convert loss to float at val sample {i}, skipping (loss: {loss}, error: {e})")
                continue

            total_loss_sum += loss_value
            for k in metric_sums:
                v = metrics[k]
                if v == v:  # skip NaN (no valid pixels in this sample)
                    metric_sums[k] += v
            n += 1

            if save_dir is not None and i < save_n:
                np.save(save_dir / f"val_sample{i}_depth_mm.npy", depth_pred[0].detach().cpu().numpy())
                np.save(save_dir / f"val_sample{i}_gt_depth_mm.npy", prepared["gt_depth"][0].detach().cpu().numpy())
                np.save(save_dir / f"val_sample{i}_mask_logit.npy", mask_logit[0].detach().cpu().numpy())
    encoder.train()
    decoder.train()
    if n == 0:
        return float("nan"), {"absrel": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    avg_loss = total_loss_sum / n
    avg_metrics = {k: v / n for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


def find_latest_checkpoint(checkpoints_dir):
    """Return the path to the checkpoint with the highest global_iter in
    checkpoints_dir, or None if the directory doesn't exist or is empty.
    """
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.is_dir():
        return None
    candidates = sorted(ckpt_dir.glob("*.pt"))
    if not candidates:
        return None
    best_path, best_iter = None, -1
    for path in candidates:
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            continue
        it = int(ckpt.get("global_iter", -1))
        if it > best_iter:
            best_iter = it
            best_path = path
    return str(best_path) if best_path is not None else None

def _batch_strings(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _collate_preprocessed_batch(batch):
    """Collate the single-resolution batch without shared tensor storage."""
    collated = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if all(torch.is_tensor(value) for value in values):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = torch.utils.data.default_collate(values)
    return collated


def split_by_frame(samples, val_fraction):
    samples = sorted(
        samples,
        key=lambda s: (s["scene"], s["cam"], int(s["frame"]))
    )
    n_val = max(1, int(len(samples) * val_fraction))
    val = samples[-n_val:]
    train = samples[:-n_val]
    return train, val, n_val

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {DEVICE}")
    ds = PreprocessedHypersimDataset(data_dir=args.data_dir)
    print(f"Found {len(ds)} frames")

    # Setup logging (CSV or TensorBoard)
    if args.use_tensorboard:
        if SummaryWriter is None:
            raise ImportError("TensorBoard is not installed. Please install tensorboard or disable --use-tensorboard.")
        tb_log_dir = Path(args.out_dir) / "tb_logs"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logging enabled. Logs will be saved to: {tb_log_dir}")
        csv_path = None  # Disable CSV logging when using TensorBoard
    else:
        # Setup CSV logging
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "metrics.csv"
        if not csv_path.exists():
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'phase', 'loss', 'absrel', 'mae', 'rmse', 'delta1', 'delta2', 'delta3', 'delta_1025', 'iter_time'])

    # Convert dataset to list for train/val split (no shuffle - use original order).
    indices = list(range(len(ds)))

    # Split into train and validation
    val_fraction = args.val_fraction
    n_val = max(1, int(len(indices) * val_fraction))
    val_indices = indices[-n_val:]
    train_indices = indices[:-n_val]

    # Create train and validation datasets
    train_ds = torch.utils.data.Subset(ds, train_indices)
    val_ds = torch.utils.data.Subset(ds, val_indices)

    print(f"Train: {len(train_ds)} samples")
    print(f"Val:   {len(val_ds)} samples")
    if not train_ds:
        raise ValueError("Training dataset is empty after the validation split")

    # Epoch tracking variables (initialized after train_ds is known)
    current_epoch = 0
    samples_in_current_epoch = 0
    samples_per_epoch = len(train_ds)

    print("Building model...")
    encoder, decoder = build_model()

    # clamp cls_token gradients to prevent NaN from attention instability
    def _clip_cls_token(grad):
        return torch.nan_to_num(grad, nan=0.0).clamp(-0.01, 0.01)

    # Handle DataParallel wrapping for cls_token access
    encoder_to_check = encoder.module if isinstance(encoder, torch.nn.DataParallel) else encoder
    if hasattr(encoder_to_check.model_rgb, 'cls_token'):
        encoder_to_check.model_rgb.cls_token.register_hook(_clip_cls_token)
    if hasattr(encoder_to_check.model_depth, 'cls_token'):
        encoder_to_check.model_depth.cls_token.register_hook(_clip_cls_token)

    if args.freeze_encoder:
        print("Freezing encoder weights.")
        for p in encoder.parameters():
            p.requires_grad = False

    # Create optimizer & scheduler once
    trainable_enc = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": trainable_enc, "lr": args.enc_lr},
        {"params": decoder.parameters(), "lr": args.dec_lr},
    ], weight_decay=args.weight_decay)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_iters
    )
    decay_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_steps, gamma=0.5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[args.warmup_iters]
    )

    global_iter = 0
    resume_step = 0

    # Handle overfitting mode
    if args.overfit:
        print("OVERFIT MODE: Training on single sample")
        # Use the full dataset for overfitting (can pick any sample)
        data_dir = args.data_dir

        # Get the specific sample to overfit on
        sample_idx = min(args.overfit_sample_idx, len(ds) - 1)
        sample = ds[sample_idx]
        print(f"Overfitting on sample index {sample_idx}: {sample['scene']}/{sample['cam']}/frame.{sample['frame']}")
        print(f"Sample keys: {list(sample.keys())}")
        print(f"RGB shape: {sample['rgb'].shape}")
        print(f"GT depth shape: {sample['gt_depth'].shape}")
        print(f"GT mask shape: {sample['gt_mask'].shape}")
        print(f"Depth filled mm shape: {sample['depth_filled_mm'].shape}")
        print(f"Valid mask shape: {sample['valid_mask'].shape}")

        # Create single-sample datasets
        overfit_train_ds = torch.utils.data.Subset(ds, [sample_idx])
        overfit_val_ds = torch.utils.data.Subset(ds, [sample_idx])  # Use same sample for validation

        optimizer.param_groups[0]['lr'] = args.enc_lr  # Encoder LR
        optimizer.param_groups[1]['lr'] = args.dec_lr  # Decoder LR

        # Run overfitting on a single frame
        print("Starting overfitting...")

        # Extract + preprocess the sample once (padding, intrinsics, normalization)
        prepared = prepare_sample(sample, DEVICE)

        print(f"After padding:")
        prepared_sample = prepared
        print(f"  rgb_padded: {prepared_sample['rgb_p'].shape}")
        print(f"  depth_input_padded: {prepared_sample['depth_input_p'].shape}")
        print(f"  Patches: {prepared_sample['h_patch']}x{prepared_sample['w_patch']}")
        print(f"  GT depth: {prepared_sample['gt_depth'].shape}")
        print(f"  GT mask: {prepared_sample['gt_mask'].shape}")
        print(f"  Valid mask: {prepared_sample['valid_mask'].shape}")
        print(f"  Alpha: {prepared_sample['alpha']:.6f}, Beta: {prepared_sample['beta']:.6f}")
        print(f"  Original size: {prepared_sample['orig_h']}x{prepared_sample['orig_w']}")
        print(f"  Padded size: {prepared_sample['padded_h']}x{prepared_sample['padded_w']}")
        print(f"  FX: {prepared_sample['fx']:.2f}, FY: {prepared_sample['fy']:.2f}, CX: {prepared_sample['cx']:.2f}, CY: {prepared_sample['cy']:.2f}")

        all_params = list(encoder.parameters()) + list(decoder.parameters())

        overfit_pred_dir = Path(args.out_dir) / "predictions" / "overfit"
        overfit_pred_dir.mkdir(parents=True, exist_ok=True)
        np.save(overfit_pred_dir / "gt_depth_mm.npy", prepared_sample["gt_depth"][0].detach().cpu().numpy())
        save_every = (
            args.overfit_save_every
            if args.overfit_save_every > 0
            else max(1, args.overfit_iters // 10)
        )

        # Training loop
        for it in range(1, args.overfit_iters + 1):
            optimizer.zero_grad()

            loss, parts, depth_pred, mask_logit = forward_and_loss(encoder, decoder, prepared, args)

            # Skip iteration if loss is NaN or infinite (robust check for tensor/non-tensor)
            is_invalid = False
            if torch.is_tensor(loss):
                if torch.isnan(loss) or torch.isinf(loss):
                    is_invalid = True
            else:
                # Handle non-tensor losses (numpy arrays, Python scalars, etc.)
                try:
                    loss_val = float(loss)
                    if math.isnan(loss_val) or math.isinf(loss_val):
                        is_invalid = True
                except (ValueError, TypeError):
                    # If we can't convert to float, treat as invalid
                    is_invalid = True

            if is_invalid:
                print(f"  WARNING: Invalid loss at iter {it}, skipping (loss type: {type(loss)}, value: {loss})")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip)
            optimizer.step()

            # Safely convert loss to Python float
            try:
                loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
            except (ValueError, TypeError) as e:
                print(f"  WARNING: Could not convert loss to float at iter {it}, skipping (loss: {loss}, error: {e})")
                optimizer.zero_grad()
                continue
            if it == 1:
                first_loss = loss_val

            # Log to CSV or TensorBoard every log_every iterations
            if it % args.log_every == 0:
                with torch.no_grad():
                    metrics = compute_batch_metrics(depth_pred, prepared)
                if args.use_tensorboard and writer is not None:
                    writer.add_scalar('overfit_train/loss', loss_val, it)
                    if metrics:
                        for key, value in metrics.items():
                            if key in ['mae', 'rmse']:
                                value = value * 1000.0
                                value = round(value, 1)
                            writer.add_scalar(f'overfit_train/{key}', value, it)
                else:
                    log_to_csv(csv_path, it, "overfit_train", loss_val, metrics)

            # Print detailed loss components every print_every iterations
            if it % args.print_every == 0 or it == args.overfit_iters:
                print(f"  Loss components: {parts}")

            if it % save_every == 0 or it == args.overfit_iters:
                np.save(overfit_pred_dir / f"pred_depth_mm_iter{it}.npy", depth_pred[0].detach().cpu().numpy())

            if it % args.print_every == 0 or it == args.overfit_iters:
                with torch.no_grad():
                    metrics = compute_batch_metrics(depth_pred, prepared)

                print(f"  iter {it:4d}/{args.overfit_iters}  loss={loss_val:.5f}  "
                      f"absrel={metrics['absrel']:.4f}  mae={metrics['mae']:.1f}mm")

        print(f"Saved overfit predictions every {save_every} iters to {overfit_pred_dir}")
        print("OVERFIT COMPLETE!")
        return  # Exit after overfitting

    if args.resume:
        resume_path = args.resume
    else:
        resume_path = find_latest_checkpoint(args.checkpoints_dir)
        if resume_path is not None:
            print(f"Auto-resume: found checkpoint {resume_path}")

    if resume_path and not args.dry_run:
        ckpt = torch.load(resume_path, map_location=DEVICE)
        # Handle DataParallel vs non-DataParallel checkpoint loading
        encoder_state = ckpt["encoder_state_dict"]
        decoder_state = ckpt["decoder_state_dict"]

        # If loading into DataParallel model but checkpoint doesn't have 'module.' prefix
        if isinstance(encoder, torch.nn.DataParallel) and not any(key.startswith('module.') for key in encoder_state.keys()):
            encoder_state = {'module.' + k: v for k, v in encoder_state.items()}
        # If loading into non-DataParallel model but checkpoint has 'module.' prefix
        elif not isinstance(encoder, torch.nn.DataParallel) and any(key.startswith('module.') for key in encoder_state.keys()):
            encoder_state = {k.replace('module.', '', 1): v for k, v in encoder_state.items()}

        # Same for decoder
        if isinstance(decoder, torch.nn.DataParallel) and not any(key.startswith('module.') for key in decoder_state.keys()):
            decoder_state = {'module.' + k: v for k, v in decoder_state.items()}
        elif not isinstance(decoder, torch.nn.DataParallel) and any(key.startswith('module.') for key in decoder_state.keys()):
            decoder_state = {k.replace('module.', '', 1): v for k, v in decoder_state.items()}

        encoder.load_state_dict(encoder_state)
        decoder.load_state_dict(decoder_state)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_iter = int(ckpt.get("global_iter", 0))
        resume_step = int(ckpt.get("train_step", 0))
        print(f"Resumed from {resume_path} at global_iter={global_iter}, train_step={resume_step}")

    # Keep TensorBoard's x-axis local to this invocation while checkpoint
    # iteration counters continue from the resumed training position.
    tensorboard_step_offset = resume_step
    if args.use_tensorboard and tensorboard_step_offset:
        print(
            f"TensorBoard step reset: displaying resumed iterations from 0 "
            f"(checkpoint step {tensorboard_step_offset})"
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=_collate_preprocessed_batch,
    )

    batches_per_epoch = len(train_loader)
    if batches_per_epoch == 0:
        raise ValueError(
            f"Training data has no full batches for batch size {args.batch_size}"
        )

    # Handle epoch-based training
    if args.epochs > 0:
        # Calculate total iterations based on epochs
        samples_per_epoch = batches_per_epoch * args.batch_size
        iterations_per_epoch = batches_per_epoch
        total_iters = args.epochs * iterations_per_epoch
        print(f"Epoch-based training: {args.epochs} epochs × {iterations_per_epoch} iterations/epoch = {total_iters} total iterations")
    else:
        # Use existing iteration-based logic (backward compatibility)
        total_iters = batches_per_epoch
        print(f"Dataset: {len(train_ds)} samples, Batch size: {args.batch_size} → {total_iters} total iterations")

    val_every = max(1, int(total_iters * args.val_every_fraction))
    val_every_percent = (val_every / total_iters) * 100
    print(f"Validation every {val_every} iterations ({val_every_percent:.1f}%)")
    print(f"Will validate at iterations: {list(range(val_every, total_iters+1, val_every))[:10]}{'...' if total_iters > val_every*10 else ''}")

    ckpt_dir_path = Path(args.checkpoints_dir)
    if args.save_weights:
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)
    ckpt_interval = (
        max(1, int(total_iters * args.save_checkpoint_fraction))
        if args.save_checkpoint_fraction > 0
        else None
    )
    print(f"Checkpoint directory: {ckpt_dir_path.resolve()}")
    if ckpt_interval is None:
        print("Periodic checkpointing disabled")
    else:
        print(
            f"Periodic checkpoints every {ckpt_interval} iterations "
            f"(fraction={args.save_checkpoint_fraction})"
        )

    # Initialize best validation metric tracking
    best_val_delta1 = -1.0  # delta1 is between 0 and 1, so -1 is worse than any possible value

    all_params = list(encoder.parameters()) + list(decoder.parameters())

    # Handle resumption: determine how many samples to skip
    samples_to_skip = resume_step * args.batch_size

    encoder.train()
    decoder.train()

    optimizer.zero_grad()
    accum_count = 0
    accum_loss_sum = 0.0
    accum_time = 0.0
    last_iter_time = 0.0
    consecutive_invalid_losses = 0
    max_consecutive_invalid_losses = 10  # Force step after this many consecutive invalid losses
    total_invalid_losses = 0  # Track total NaN/invalid losses encountered

    # Track global sample count across epochs
    global_samples_seen = 0
    step_start_time = None

    # Epoch tracking
    current_epoch = 0
    samples_in_current_epoch = 0
    samples_per_epoch = len(train_ds)

    # Handle resumption for epoch tracking
    if args.epochs > 0 and resume_step > 0:
        # Calculate which epoch we should be in and position within that epoch
        iterations_per_epoch = max(1, samples_per_epoch // args.batch_size)
        current_epoch = resume_step // iterations_per_epoch
        samples_in_current_epoch = (resume_step % iterations_per_epoch) * args.batch_size
        # Ensure we don't exceed samples per epoch
        samples_in_current_epoch = min(samples_in_current_epoch, samples_per_epoch)

    train_iter = iter(train_loader)
    first_training_sample_reported = False
    first_batch_debug_reported = False
    print(f"--- Starting training loop ---")
    while global_iter < total_iters:
        timing_start = None
        fetch_start = time.perf_counter() if args.timing else None
        try:
            sample = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            sample = next(train_iter)

        if args.timing:
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            fetch_time = time.perf_counter() - fetch_start
            timing_start = time.perf_counter()

        global_samples_seen += args.batch_size

        # Skip samples already processed (for resumption)
        if global_samples_seen <= samples_to_skip:
            continue

        # Start timing for this optimization step if we haven't already
        if accum_count == 0 and step_start_time is None:
            step_start_time = time.perf_counter()

        if not first_training_sample_reported:
            dataset_index = int(sample["dataset_index"][0])
            frame = int(sample["frame"][0])
            print(
                f"First training sample: index={dataset_index} "
                f"(frame_{frame})"
            )
            first_training_sample_reported = True

        if args.dry_run:
            with torch.no_grad():
                prepared = prepare_sample(sample, DEVICE)
                loss, parts, depth_pred, mask_logit = forward_and_loss(encoder, decoder, prepared, args)
            if args.timing:
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                compute_time = time.perf_counter() - timing_start
                print(
                    f"[timing] iter={global_iter} fetch={fetch_time:.3f}s "
                    f"compute={compute_time:.3f}s total={fetch_time + compute_time:.3f}s"
                )
            if not first_batch_debug_reported:
                print(
                    "[batch-debug] "
                    f"raw_rgb={tuple(sample['rgb'].shape)} "
                    f"prepared_rgb={tuple(prepared['rgb_p'].shape)} "
                    f"prepared_depth={tuple(prepared['depth_input_p'].shape)} "
                    f"prediction={tuple(depth_pred.shape)} "
                    f"loss={loss.item():.6f}"
                )
                first_batch_debug_reported = True
            print(f"[dry-run] batch {global_samples_seen//args.batch_size}/{(len(train_ds)+args.batch_size-1)//args.batch_size}  loss={loss.item():.5f}")
            if global_samples_seen - samples_to_skip >= args.batch_size:
                break
            continue

        prepared = prepare_sample(sample, DEVICE)
        loss, parts, depth_pred, mask_logit = forward_and_loss(encoder, decoder, prepared, args)
        if not first_batch_debug_reported:
            print(
                "[batch-debug] "
                f"raw_rgb={tuple(sample['rgb'].shape)} "
                f"prepared_rgb={tuple(prepared['rgb_p'].shape)} "
                f"prepared_depth={tuple(prepared['depth_input_p'].shape)} "
                f"prediction={tuple(depth_pred.shape)} "
                f"mask_logits={tuple(mask_logit.shape)} "
                f"loss={loss.item():.6f} "
                f"loss_requires_grad={loss.requires_grad}"
            )
            first_batch_debug_reported = True
        # log raw loss every sample for first 20 samples
        if global_samples_seen <= 20:
            loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
            print(f"  [diag] sample {global_samples_seen} raw loss={loss_val:.4f} parts={parts}")

        # diagnose NaN source
        if torch.is_tensor(loss) and torch.isnan(loss):
            print(f"  [diag] depth_input min={prepared['depth_input_p'].min():.4f} max={prepared['depth_input_p'].max():.4f}")
            print(f"  [diag] pred NaN={torch.isnan(depth_pred).any()} Inf={torch.isinf(depth_pred).any()}")
            for name, p in encoder.named_parameters():
                if torch.isnan(p).any():
                    print(f"  [diag] NaN weight: {name}")
                    break

        # Skip iteration if loss is NaN or infinite (robust check for tensor/non-tensor)
        is_invalid = False
        if torch.is_tensor(loss):
            if torch.isnan(loss) or torch.isinf(loss):
                is_invalid = True
        else:
            # Handle non-tensor losses (numpy arrays, Python scalars, etc.)
            try:
                loss_val = float(loss)
                if math.isnan(loss_val) or math.isinf(loss_val):
                    is_invalid = True
            except (ValueError, TypeError):
                # If we can't convert to float, treat as invalid
                is_invalid = True

        if is_invalid:
            total_invalid_losses += 1
            global_iter += 1
            resume_step = global_iter
            # Log detailed information about the problematic sample
            print(f"  WARNING: Invalid loss at iter {global_iter}, skipping (loss type: {type(loss)}, value: {loss})")
            print(
                f"    Batch info: dataset_index={sample['dataset_index']}, "
                f"scene={sample['scene']}, cam={sample['cam']}, frame={sample['frame']}"
            )
            print(f"    GT depth range: {prepared['gt_depth'].min():.1f}-{prepared['gt_depth'].max():.1f}mm")
            print(f"    GT mask ratio: {prepared['gt_mask'].float().mean():.4f} ({100*prepared['gt_mask'].float().mean():.2f}%)")
            print(f"    Valid mask ratio: {prepared['valid_mask'].float().mean():.4f} ({100*prepared['valid_mask'].float().mean():.2f}%)")
            # Handle combined mask calculation to avoid CUDA bitwise_and issues
            combined_mask = prepared['gt_mask'].bool() & prepared['valid_mask'].bool()
            print(f"    Combined mask ratio: {combined_mask.float().mean():.4f} ({100*combined_mask.float().mean():.2f}%)")
            print(f"    Pred depth range: {depth_pred.min():.1f}-{depth_pred.max():.1f}mm")
            if torch.is_tensor(depth_pred):
                print(f"    Pred has NaN: {torch.isnan(depth_pred).any()}, Inf: {torch.isinf(depth_pred).any()}")

            consecutive_invalid_losses += 1

            # If we get too many consecutive invalid losses, force a step to prevent getting stuck
            if consecutive_invalid_losses >= max_consecutive_invalid_losses:
                print(f"  WARNING: Too many consecutive invalid losses ({consecutive_invalid_losses}), forcing optimization step")
                # Zero gradients and take a step to prevent getting stuck in accumulation loop
                if accum_count > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                avg_train_loss = accum_loss_sum / max(accum_count, 1)
                print(f"  FORCED STEP: iter {global_iter}  train_loss={avg_train_loss:.5f}")

                accum_loss_sum = 0.0
                accum_count = 0
                consecutive_invalid_losses = 0
                # Reset step_start_time for the next step
                step_start_time = None
                # Update epoch tracking
                if args.epochs > 0:
                    samples_in_current_epoch += args.batch_size
                    if samples_in_current_epoch >= samples_per_epoch:
                        current_epoch += 1
                        samples_in_current_epoch -= samples_per_epoch
                        print(f"Completed epoch {current_epoch}")
            if args.timing:
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                print(
                    f"[timing] iter={global_iter} fetch={fetch_time:.3f}s "
                    f"compute={time.perf_counter() - timing_start:.3f}s "
                    "status=skipped"
                )
            continue

        consecutive_invalid_losses = 0
        loss.backward()

        # Safely convert loss to Python float for accumulation
        try:
            loss_value = loss.item() if torch.is_tensor(loss) else float(loss)
        except (ValueError, TypeError) as e:
            print(f"  WARNING: Could not convert loss to float at iter {global_iter+1}, skipping (loss: {loss}, error: {e})")
            optimizer.zero_grad()
            continue

        accum_loss_sum += loss_value
        accum_count += args.batch_size

        if accum_count == args.batch_size:
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip)
            if global_iter % 10 == 0:
                print(f"  [diag] grad_norm={grad_norm:.4f}")
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if args.timing:
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                compute_time = time.perf_counter() - timing_start
                print(
                    f"[timing] iter={global_iter + 1} fetch={fetch_time:.3f}s "
                    f"compute={compute_time:.3f}s total={fetch_time + compute_time:.3f}s"
                )

            # check for NaN weights after step
            for name, p in encoder.named_parameters():
                if torch.isnan(p).any():
                    raise RuntimeError(f"NaN weights detected in {name} at iter {global_iter+1}")

            global_iter += 1
            resume_step = global_iter
            avg_train_loss = accum_loss_sum / accum_count
            accum_loss_sum = 0.0
            accum_count = 0

            # Calculate step time
            step_time = time.perf_counter() - step_start_time if step_start_time is not None else 0.0
            # Reset step_start_time for the next step
            step_start_time = None

            # Log training metrics to CSV or TensorBoard every log_every iterations
            if global_iter % args.log_every == 0:
                with torch.no_grad():
                    metrics = compute_batch_metrics(depth_pred, prepared)
                if args.use_tensorboard and writer is not None:
                    tensorboard_step = global_iter - tensorboard_step_offset
                    writer.add_scalar('train/loss', avg_train_loss, tensorboard_step)
                    writer.add_scalar('train/invalid_loss_count', total_invalid_losses, tensorboard_step)
                    writer.add_scalar('train/invalid_loss_rate', total_invalid_losses / max(global_iter, 1), tensorboard_step)
                    if metrics:
                        for key, value in metrics.items():
                            if key in ['mae', 'rmse']:
                                value = value * 1000.0
                                value = round(value, 1)
                            writer.add_scalar(f'train/{key}', value, tensorboard_step)
                else:
                    # Convert mae and rmse from meters to millimeters and round to 1 decimal place for CSV logging
                    metrics_mm = metrics.copy()
                    metrics_mm['mae'] = round(metrics['mae'] * 1000.0, 1)
                    metrics_mm['rmse'] = round(metrics['rmse'] * 1000.0, 1)
                    log_to_csv(csv_path, global_iter, "train", avg_train_loss, metrics_mm)

            # Save periodic checkpoint if fraction is set (independent of validation)
            if ckpt_interval is not None and global_iter % ckpt_interval == 0:
                ckpt_path = ckpt_dir_path / f"ckpt_iter{global_iter}.pt"
                save_checkpoint(
                    ckpt_path, train_step=global_iter, global_iter=global_iter,
                    encoder=encoder, decoder=decoder, optimizer=optimizer, scheduler=scheduler,
                    train_loss=avg_train_loss,
                )
                print(f"  [periodic checkpoint saved: {ckpt_path}] (iter {global_iter})")

            if global_iter % max(1, val_every // 5) == 0 or global_iter == total_iters:
                # Calculate next validation iteration
                next_val = ((global_iter // val_every) + 1) * val_every
                if next_val > total_iters:
                    next_val_str = "end"
                else:
                    next_val_str = f"{next_val}"
                print(f"iter {global_iter:5d}/{total_iters}  train_loss={avg_train_loss:.5f}  [next val at iter {next_val_str}]")

            if global_iter % val_every == 0 or global_iter == total_iters:
                print(f"--- Running validation at iter {global_iter} ---")
                pred_save_dir = Path(args.out_dir) / "predictions" / f"iter{global_iter}"
                val_loss, val_metrics = run_validation(encoder, decoder, val_ds, args, save_dir=pred_save_dir)
                print(f"  [val predictions saved: {pred_save_dir}]")

                # Log validation metrics to CSV or TensorBoard (convert mae and rmse to millimeters and round to 1 decimal place)
                if args.use_tensorboard and writer is not None:
                    tensorboard_step = global_iter - tensorboard_step_offset
                    writer.add_scalar('val/loss', val_loss, tensorboard_step)
                    if val_metrics:
                        for key, value in val_metrics.items():
                            if key in ['mae', 'rmse']:
                                value = value * 1000.0
                                value = round(value, 1)
                            writer.add_scalar(f'val/{key}', value, tensorboard_step)
                else:
                    val_metrics_mm = val_metrics.copy()
                    val_metrics_mm['mae'] = round(val_metrics['mae'] * 1000.0, 1)
                    val_metrics_mm['rmse'] = round(val_metrics['rmse'] * 1000.0, 1)
                    log_to_csv(csv_path, global_iter, "val", val_loss, val_metrics_mm)

                # Track best validation delta1 and save best checkpoint
                current_val_delta1 = val_metrics["delta1"]
                is_best = current_val_delta1 > best_val_delta1
                if is_best:
                    print(f"  [NEW BEST] val_delta1 improved from {best_val_delta1:.4f} to {current_val_delta1:.4f}")
                    best_val_delta1 = current_val_delta1
                    # Remove previous best checkpoint if it exists
                    best_ckpt_path = ckpt_dir_path / "ckpt_best.pt"
                    if best_ckpt_path.exists():
                        best_ckpt_path.unlink()
                        print(f"  [REMOVED OLD BEST] {best_ckpt_path}")
                    # Save new best checkpoint
                    best_ckpt_path = ckpt_dir_path / "ckpt_best.pt"
                    save_checkpoint(
                        best_ckpt_path, train_step=global_iter, global_iter=global_iter,
                        encoder=encoder, decoder=decoder, optimizer=optimizer, scheduler=scheduler,
                        train_loss=avg_train_loss, val_loss=val_loss, val_absrel=val_metrics["absrel"],
                        val_delta1=val_metrics["delta1"], val_delta_1025=val_metrics["delta_1025"],
                    )
                    print(f"  [BEST CHECKPOINT SAVED] {best_ckpt_path}")
                else:
                    print(f"  [NOT BEST] val_delta1={current_val_delta1:.4f} (best so far: {best_val_delta1:.4f})")

                if args.save_weights:
                    ckpt_path = ckpt_dir_path / f"ckpt_iter{global_iter}.pt"
                    save_checkpoint(
                        ckpt_path, train_step=global_iter, global_iter=global_iter,
                        encoder=encoder, decoder=decoder, optimizer=optimizer, scheduler=scheduler,
                        train_loss=avg_train_loss, val_loss=val_loss, val_absrel=val_metrics["absrel"],
                        val_delta1=val_metrics["delta1"], val_delta_1025=val_metrics["delta_1025"],
                    )
                    print(f"  [checkpoint saved: {ckpt_path}]")

            if global_iter >= total_iters:
                break

            # Update epoch tracking
            if args.epochs > 0:
                samples_in_current_epoch += args.batch_size
                if samples_in_current_epoch >= samples_per_epoch:
                    current_epoch += 1
                    samples_in_current_epoch -= samples_per_epoch
                    print(f"Completed epoch {current_epoch}")

    print("TRAINING COMPLETE!")
    # Close TensorBoard writer if used
    if args.use_tensorboard and writer is not None:
        writer.close()
        print("TensorBoard writer closed.")

    if args.save_weights:
        final_ckpt_path = ckpt_dir_path / "ckpt_final.pt"
        save_checkpoint(
            final_ckpt_path, train_step=global_iter, global_iter=global_iter,
            encoder=encoder, decoder=decoder, optimizer=optimizer, scheduler=scheduler,
        )
        print(f"[final checkpoint saved: {final_ckpt_path}]")


def test_model(args):
    """Run inference on test data and save predictions."""
    print("--- Running test mode ---")

    # Validate arguments
    if not args.checkpoint:
        raise ValueError("--checkpoint is required when using --test")

    # Setup device
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for testing!")
            DEVICE = torch.device("cuda:0")
        else:
            DEVICE = torch.device("cuda:0")
            print("Using single GPU for testing.")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU for testing.")

    # Load test dataset
    test_dataset = PreprocessedHypersimDataset(data_dir=args.test_dir)
    print(f"Found {len(test_dataset)} test samples")

    # Build model
    print("Building model...")
    model_rgb = load_backbone()
    model_depth = load_backbone()
    encoder = DualBranchEncoder(model_rgb, model_depth)
    decoder = DPTDecoder()

    # Wrap with DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)

    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)

    # Handle DataParallel vs non-DataParallel checkpoint loading
    encoder_state = checkpoint["encoder_state_dict"]
    decoder_state = checkpoint["decoder_state_dict"]

    if isinstance(encoder, torch.nn.DataParallel) and not any(key.startswith('module.') for key in encoder_state.keys()):
        encoder_state = {'module.' + k: v for k, v in encoder_state.items()}
    elif not isinstance(encoder, torch.nn.DataParallel) and any(key.startswith('module.') for key in encoder_state.keys()):
        encoder_state = {k.replace('module.', '', 1): v for k, v in encoder_state.items()}

    # Same for decoder
    if isinstance(decoder, torch.nn.DataParallel) and not any(key.startswith('module.') for key in decoder_state.keys()):
        decoder_state = {'module.' + k: v for k, v in decoder_state.items()}
    elif not isinstance(decoder, torch.nn.DataParallel) and any(key.startswith('module.') for key in decoder_state.keys()):
        decoder_state = {k.replace('module.', '', 1): v for k, v in decoder_state.items()}

    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)
    print("Checkpoint loaded successfully")

    # Set to evaluation mode
    encoder.eval()
    decoder.eval()

    # Setup save directory
    save_dir = Path(args.test_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving predictions to {save_dir}")

    # Run inference
    total_loss_sum = 0.0
    metric_sums = {"absrel": 0.0, "mae": 0.0, "rmse": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0, "delta_1025": 0.0}
    n = 0

    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            prepared = prepare_sample(sample, DEVICE)
            loss, parts, depth_pred, mask_logit = forward_and_loss(encoder, decoder, prepared, args)

            # Skip iteration if loss is NaN or infinite (robust check for tensor/non-tensor)
            is_invalid = False
            if torch.is_tensor(loss):
                if torch.isnan(loss) or torch.isinf(loss):
                    is_invalid = True
            else:
                # Handle non-tensor losses (numpy arrays, Python scalars, etc.)
                try:
                    loss_val = float(loss)
                    if math.isnan(loss_val) or math.isinf(loss_val):
                        is_invalid = True
                except (ValueError, TypeError):
                    # If we can't convert to float, treat as invalid
                    is_invalid = True

            if is_invalid:
                print(f"  WARNING: Invalid loss at test sample {i}, skipping (loss type: {type(loss)}, value: {loss})")
                continue

            metrics = compute_batch_metrics(depth_pred, prepared)

            # Safely convert loss to Python float for accumulation
            try:
                loss_value = loss.item() if torch.is_tensor(loss) else float(loss)
            except (ValueError, TypeError) as e:
                print(f"  WARNING: Could not convert loss to float at test sample {i}, skipping (loss: {loss}, error: {e})")
                continue

            total_loss_sum += loss_value
            for k in metric_sums:
                v = metrics[k]
                if v == v:  # skip NaN (no valid pixels in this sample)
                    metric_sums[k] += v
            n += 1

            # Save predictions and visualizations for first N samples
            if i < args.test_save_n:
                # Save .npy files
                np.save(save_dir / f"test_sample{i}_depth_mm.npy", depth_pred[0].detach().cpu().numpy())
                np.save(save_dir / f"test_sample{i}_gt_depth_mm.npy", prepared["gt_depth"][0].detach().cpu().numpy())
                np.save(save_dir / f"test_sample{i}_mask_logit.npy", mask_logit[0].detach().cpu().numpy())

                # Generate visualization if matplotlib is available
                if plt is not None:
                    try:
                        # Prepare data
                        depth_pred_np = depth_pred[0].detach().cpu().numpy().squeeze()
                        gt_depth_np = prepared["gt_depth"][0].detach().cpu().numpy().squeeze()
                        mask_logit_np = mask_logit[0].detach().cpu().numpy().squeeze()
                        valid_mask_np = prepared["valid_mask"][0].detach().cpu().numpy().astype(bool)

                        error_map = np.abs(gt_depth_np - depth_pred_np)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        im = ax.imshow(gt_depth_np, cmap='viridis')
                        ax.set_title('GT Depth (mm)')
                        plt.colorbar(im, ax=ax)
                        fig.tight_layout()
                        fig.savefig(save_dir / f"test_sample{i}_gt.png", dpi=150)
                        plt.close(fig)

                        fig, ax = plt.subplots(figsize=(6, 5))
                        im = ax.imshow(depth_pred_np, cmap='viridis')
                        ax.set_title('Predicted Depth (mm)')
                        plt.colorbar(im, ax=ax)
                        fig.tight_layout()
                        fig.savefig(save_dir / f"test_sample{i}_pred.png", dpi=150)
                        plt.close(fig)

                        fig, ax = plt.subplots(figsize=(6, 5))
                        im = ax.imshow(error_map, cmap='hot')
                        ax.set_title('Abs Error (mm)')
                        plt.colorbar(im, ax=ax)
                        fig.tight_layout()
                        fig.savefig(save_dir / f"test_sample{i}_error.png", dpi=150)
                        plt.close(fig)
                    except Exception as e:
                        print(f"  WARNING: Failed to generate visualization for sample {i}: {e}")
                else:
                    # Warn only once that matplotlib is not available
                    if not hasattr(test_model, "_matplotlib_warned"):
                        print("  WARNING: matplotlib not installed, skipping visualization")
                        test_model._matplotlib_warned = True

    # Calculate and print results
    if n == 0:
        print("No valid samples found for testing")
        return float("nan"), {"absrel": float("nan"), "mae": float("nan"), "rmse": float("nan")}

    avg_loss = total_loss_sum / n
    avg_metrics = {k: v / n for k, v in metric_sums.items()}

    # Convert mae and rmse from meters to millimeters for display
    print(f"Test Results ({n} valid samples):")
    print(f"  Loss: {avg_loss:.5f}")
    print(f"  AbsRel: {avg_metrics['absrel']:.4f}")
    print(f"  MAE: {avg_metrics['mae'] * 1000.0:.1f} mm")
    print(f"  RMSE: {avg_metrics['rmse'] * 1000.0:.1f} mm")
    print(f"  Delta1: {avg_metrics['delta1']:.4f}")
    print(f"  Delta2: {avg_metrics['delta2']:.4f}")
    print(f"  Delta3: {avg_metrics['delta3']:.4f}")
    print(f"  Delta_1025: {avg_metrics['delta_1025']:.4f}")
    print(f"Predictions saved to: {save_dir}")

    return avg_loss, avg_metrics


if __name__ == "__main__":
    args = parse_args()

    if args.test:
        # Run test mode
        test_model(args)
    else:
        # Run normal training
        main()
