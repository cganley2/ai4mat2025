#!/bin/bash
#SBATCH --job-name=mace-load-ran
#SBATCH --partition=normal_q
#SBATCH --account=aiaa2026
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --mail-user=cganley2@vt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /home/cganley2/.bashrc
conda activate mlip

python load-model-predictions.py
