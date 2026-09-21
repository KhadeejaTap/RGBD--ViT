#!/bin/bash
#SBATCH --mail-user=ktapkirw@cougarnet.uh.edu
#SBATCH -N 1 -n 12
#SBATCH --gpus=1
#SBATCH -t 20:00:00

# ==============================================================================
# Stage 2 Fine-Tuning: Depth Encoder Adaptation
# - Resumes from Epoch 1 best validated checkpoint (checkpoints/ckpt_best.pt at step 22,265)
# - Trains for 2 additional epochs (9,130 iterations, total_iter 22,265 -> 31,395)
# - 98% train / 2% validation split (--val-fraction 0.02)
# - Validates 2 times (--val-every-fraction 0.5): at iter 26,830 and 31,395
# - 3x L1 loss weight (--w-l1 3.0) for prioritized metric depth accuracy
# - Gentler gradient clipping (--grad-clip 0.5) with per-module independence
# - Batched validation loop (2.5 min vs 40 min)
# ==============================================================================

source /home/ktapkirw/miniconda3/bin/activate && cd /home/ktapkirw/vitthing && conda activate vitthing && python3 -m main.train \
  --resume /home/ktapkirw/vitthing/checkpoints/ckpt_best.pt \
  --stage2 \
  --epochs 2 \
  --val-fraction 0.02 \
  --val-every-fraction 0.5 \
  --save-checkpoint-fraction 0.5 \
  --unfreeze-depth-from 6 \
  --layer-decay 1.0 \
  --enc_lr 1e-5 \
  --dec_lr 1e-5 \
  --grad-clip 0.5 \
  --w-l1 3.0 \
  --warmup-iters 500 \
  --batch-size 16 \
  --num-workers 6 \
  --use-tensorboard \
  --save-weights \
  --checkpoints-dir /home/ktapkirw/vitthing/checkpoints
