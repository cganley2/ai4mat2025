#!/bin/bash
#SBATCH --job-name=mace_training
#SBATCH --partition=normal_q
#SBATCH --account=aiaa2026
#SBATCH --array=1-12
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=48:00:00
#SBATCH --mail-user=cganley2@vt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

conda init
conda activate ds

# Get the directory for this task
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" dir-list.txt)

cd "$DIR"

python model.py