#!/bin/bash
#SBATCH --job-name=mace-rl1500-load
#SBATCH --partition=normal_q
#SBATCH --account=aiaa2026
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-user=cganley2@vt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /home/cganley2/.bashrc
conda activate mlip

python load-model-predictions.py
