#!/bin/bash
#SBATCH --mail-user=ktapkirw@cougarnet.uh.edu
#SBATCH -N 1 -n 12
#SBATH --gpus=1
#SBATCH -t 20:00:00
source /home/ktapkirw/miniconda3/bin/activate vitthing && cd /home/ktapkirw/vitthing && python3 -m main.train \
  --overfit \
  --overfit-iters 100000 \
  --overfit-sample-idx 1 \
  --print-every 1000 \
  --log-every 100 \
  --use-tensorboard \
  --no-save-weights \
  --overfit-save-every 1000\
  --out-dir /home/ktapkirw/vitthing/outputs/overfit_100k \
  --freeze-encoder
