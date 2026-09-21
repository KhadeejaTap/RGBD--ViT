#!/bin/bash
#SBATCH --mail-user=ktapkirw@cougarnet.uh.edu
#SBATCH -N 1
#SBATCH -t 4:00:00
source /home/ktapkirw/miniconda3/bin/activate && cd /home/ktapkirw/vitthing && conda activate vitthing && python3 -m main.preprocess
