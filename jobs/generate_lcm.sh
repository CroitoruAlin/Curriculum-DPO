#!/bin/bash -l

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

conda init
conda activate dpo
cd /home/${USER}/projects/Curriculum-DPO
echo $CONDA_DEFAULT_ENV
python data_generators/generate_data_lcm.py --save_path ./datasets --subset train --dataset drawbench --reward_fn baseline --sample_batch_size 15
