#!/bin/bash
#SBATCH --time=24:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

srun python3 main.py $1
