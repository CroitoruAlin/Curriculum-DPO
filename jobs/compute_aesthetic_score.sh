#!/bin/bash -l


echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi
source /home/fl488644/miniconda3/etc/profile.d/conda.sh
conda init bash

conda activate dpo
cd /home/${USER}/projects/Curriculum-DPO
echo $CONDA_DEFAULT_ENV 
python reward_models/compute_aesthetic_score.py --dataset_path /home/telperion/projects/Curriculum-DPO/datasets/animals/train/lcm/baseline --save_path scores/lcm/baseline_aesthetic_animals --split train
