#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem-per-cpu=4G

nvidia-smi
source /mnt/home/fmi2/miniconda3/etc/profile.d/conda.sh
conda init bash

conda activate dpo2
cd /mnt/home/fmi2/alin/public_code/Curriculum-DPO
echo $CONDA_DEFAULT_ENV
export CUDA_VISIBLE_DEVICES="0"
python data_generators/generate_data_sd.py --dataset drawbench