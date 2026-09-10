
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Directories to visualize
DIRS = [
    "",
]

for base_dir in DIRS:
    if not os.path.exists(base_dir):
        print(f"Skipping {base_dir} (not found)")
        continue

    # Match both naming patterns
    FILES = sorted(glob.glob(os.path.join(base_dir, "*_depth_*.npy")))
    OUT_DIR = os.path.join(base_dir, "vis")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n=== Visualizing {base_dir} ===")
    for f in FILES:
        depth = np.load(f).squeeze()  # (H,W)
        valid = depth > 0

        vmin = depth[valid].min() if valid.any() else 0
        vmax = depth[valid].max() if valid.any() else 1

        norm = np.clip((depth - vmin) / (vmax - vmin + 1e-6), 0, 1)
        colored = cm.turbo(norm)[..., :3]

        name = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(OUT_DIR, name + ".png")
        plt.imsave(out_path, colored)
        print(f"  {name} -> {out_path}")

print("\nDone!")
