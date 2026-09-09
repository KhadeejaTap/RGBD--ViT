#!/usr/bin/env python3
"""
Script to check resolutions of NPZ files in preprocessed dataset.
Returns the index of the first file matching target resolution.
"""

import glob
import numpy as np
from pathlib import Path
import sys

def find_first_target_index(data_dir, target_width=512, target_height=288):
    """
    Find the index of the first NPZ file matching target resolution.

    Args:
        data_dir: Path to directory containing NPZ files
        target_width: Target width (default: 512)
        target_height: Target height (default: 288)

    Returns:
        Index of first matching file (0-based), or -1 if none found.
    """
    data_path = Path(data_dir)

    # Find all NPZ files, sorted
    npz_files = sorted(list(data_path.glob("*.npz")))

    if len(npz_files) == 0:
        print(f"No NPZ files found in {data_dir}", file=sys.stderr)
        return -1

    # Check each file in order
    for idx, npz_file in enumerate(npz_files):
        try:
            with np.load(npz_file, allow_pickle=True) as data:
                # Get RGB shape: (H, W, 3)
                rgb = data["rgb"]
                if len(rgb.shape) >= 3:
                    height, width = rgb.shape[0], rgb.shape[1]
                else:
                    # Skip malformed files
                    continue

                # Check if matches target
                if width == target_width and height == target_height:
                    return idx  # 0-based index

        except Exception:
            # Skip files that can't be read
            continue

    return -1  # Not found

if __name__ == "__main__":
    # Default data directory - adjust as needed
    data_dir = "hypersim_data_simple_nanfix"

    # Allow override from command line
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    # Allow custom target resolution
    target_width = 512
    target_height = 288
    if len(sys.argv) > 3:
        target_width = int(sys.argv[2])
        target_height = int(sys.argv[3])

    index = find_first_target_index(data_dir, target_width, target_height)
    print(index)
