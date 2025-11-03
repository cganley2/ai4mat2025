#!/bin/bash
#SBATCH --job-name=sim-matrix-1500
#SBATCH --partition=normal_q
#SBATCH --account=aiaa2026
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --mail-user=cganley2@vt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

source /home/cganley2/.bashrc
conda activate ds
python mbtr-sim.py
