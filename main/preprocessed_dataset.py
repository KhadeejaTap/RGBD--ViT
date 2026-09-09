#!/usr/bin/env python3
"""
Dataset loader for preprocessed Hypersim NPZ files.
Loads preprocessed data with sensor depth, GT depth, and metadata.
"""
import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class PreprocessedHypersimDataset(Dataset):
    """
    Loads preprocessed NPZ files from hypersim_data/ (flat structure)

    Each NPZ contains:
    - rgb: (H, W, 3) float32 [0,1]
    - gt_depth: (H, W) float32 mm
    - gt_mask: (H, W) bool
    - sensor_depth: (H, W) float32 mm (padded, flood-filled)
    - sensor_mask: (H, W) bool (padded, sparse - only real measurements)
    - crop_perimeter: [y, x, h, w] in GT coords
    - sensor_crop_bounds: [minx, maxx, miny, maxy]
    - sensor_base_bounds: [minx, maxx, miny, maxy]
    - sensor_shift: [dx, dy]
    - sensor_shift_range: [dx_min, dx_max, dy_min, dy_max]
    - sensor_option: str (e.g., "160x128")
    - meta: [scene, cam, frame, stage]
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Root directory containing preprocessed NPZ files (flat structure)
        """
        self.data_dir = Path(data_dir)

        # Find all NPZ files directly in the data directory
        self.files = sorted(glob.glob(str(self.data_dir / "*.npz")))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No NPZ files found in {self.data_dir}")

        print(f"PreprocessedHypersimDataset: samples={len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = self.files[idx]
        with np.load(npz_path, allow_pickle=True) as data:
            rgb = data["rgb"].astype(np.float32)           # (H, W, 3) [0,1]
            gt_depth = data["gt_depth"].astype(np.float32)  # (H, W) mm
            gt_mask = data["gt_mask"].astype(np.bool_)      # (H, W) bool
            sensor_depth = data["sensor_depth"].astype(np.float32)  # (H, W) mm padded
            sensor_mask = data["sensor_mask"].astype(np.bool_)      # (H, W) bool padded
            crop_perimeter = data["crop_perimeter"]         # [y, x, h, w]
            sensor_option = data["sensor_option"]
            if isinstance(sensor_option, np.ndarray):
                sensor_option = sensor_option.item()
            meta = data["meta"]
            if isinstance(meta, np.ndarray):
                meta = meta.tolist()

        # Convert to tensors
        # rgb: (H, W, 3) -> (3, H, W)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().clone()

        # gt_depth: (H, W) -> (1, H, W)
        gt_depth_tensor = torch.from_numpy(gt_depth).unsqueeze(0).contiguous().clone()

        # gt_mask: (H, W) -> (1, H, W)
        gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0).contiguous().clone()

        # sensor_depth: (H, W) -> (1, H, W) - this is the flood-filled input
        sensor_depth_tensor = torch.from_numpy(sensor_depth).unsqueeze(0).contiguous().clone()

        # sensor_mask: (H, W) -> (H, W) - sparse validity mask
        sensor_mask_tensor = torch.from_numpy(sensor_mask).contiguous().clone()

        # Keep metadata tensor-backed as well so default_collate receives
        # resizable storage instead of converting a NumPy-backed array.
        crop_perimeter_tensor = torch.as_tensor(crop_perimeter).clone()

        # Intrinsics for this stage (precomputed)
        # Stage 1: 144x256, Stage 2: 288x512
        # Hypersim focal = 886.81, scale = 1/8 for stage 1, 1/5 for stage 2
        stage = meta[3]  # stage from metadata
        if stage == 1:
            scale = 1.0 / 8.0
        elif stage == 2:
            scale = 1.0 / 5.0
        else:
            scale = 1.0 / 4.0  # fallback

        focal = 886.81 * scale
        H, W = rgb.shape[:2]
        fx = fy = focal
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        return {
            "dataset_index": idx,
            "rgb": rgb_tensor,                    # (3, H, W) [0,1]
            "depth_filled_mm": sensor_depth_tensor,  # (1, H, W) mm - flood-filled sensor depth
            "valid_mask": sensor_mask_tensor,     # (H, W) bool - sparse real measurements
            "gt_depth": gt_depth_tensor,          # (1, H, W) mm
            "gt_mask": gt_mask_tensor,            # (1, H, W) bool
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "crop_perimeter": crop_perimeter_tensor,  # [y, x, h, w]
            "sensor_option": sensor_option,
            "scene": meta[0],
            "cam": meta[1],
            "frame": meta[2],
            "stage": meta[3],
        }


if __name__ == "__main__":
    # Quick test
    ds = PreprocessedHypersimDataset("../hypersim_data")
    print(f"Dataset length: {len(ds)}")

    sample = ds[55]
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.shape} {v.dtype} [{v.min():.4f}, {v.max():.4f}]")
        else:
            print(f"  {k}: {v}")
