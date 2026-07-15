#!/bin/bash -l


echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

conda activate dpo
cd /home/${USER}/projects/Curriculum-DPO
echo $CONDA_DEFAULT_ENV
wandb login 594bc31d98355581c73bec4991abf902a00c7ea3
accelerate launch train_lcm.py --data_dir datasets/drawbench/train/lcm/baseline --score_dir scores/lcm/baseline_human_drawbench  --dataset drawbench --reward_fn human_score --wandb_run_name human_drawbench_lcm --subset train --train_batch_size 10 --learning_rate 1e-4 --max_train_steps 10000
