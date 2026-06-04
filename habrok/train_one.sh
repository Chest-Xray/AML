#!/bin/bash
#SBATCH --time=20:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=8
module purge
module load CUDA-Python/12.6.2.post1-gfbf-2024a-CUDA-12.6.0
cd /scratch/$USER/AML/
source venv/bin/activate
python3 -m main $1 $2
