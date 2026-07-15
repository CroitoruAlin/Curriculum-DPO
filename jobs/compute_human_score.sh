#!/bin/bash -l

conda init
conda activate dpo
cd /home/${USER}/projects/Curriculum-DPO
echo $CONDA_DEFAULT_ENV 
python reward_models/compute_human_score.py --dataset_path /home/telperion/projects/Curriculum-DPO/datasets/drawbench/train/lcm/baseline --save_path scores/lcm/baseline_human_drawbench --split train
