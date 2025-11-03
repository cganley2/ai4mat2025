#!/bin/bash
#SBATCH --job-name=mace_training3
#SBATCH --partition=normal_q
#SBATCH --account=aiaa2026
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=12:00:00
#SBATCH --mail-user=cganley2@vt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

conda init
conda activate ds

python model_training_tutorial.py