#!/bin/bash
#SBATCH --time=05:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=8
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
cd /scratch/$USER/AML/
source venv/bin/activate
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python3 -m main $1 $2
