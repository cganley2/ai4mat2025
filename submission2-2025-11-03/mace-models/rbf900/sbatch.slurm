#!/bin/bash
#SBATCH --job-name=mace-rbf900
#SBATCH --partition=normal_q
#SBATCH --account=aiaa2026
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-user=cganley2@vt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /home/cganley2/.bashrc
conda activate mlip

python model.py

#conda activate ds
#python mbtr-sim.py
