#!/bin/bash -l

nvidia-smi
source /home/fl488644/miniconda3/etc/profile.d/conda.sh
conda init bash
cd /home/fl488644/public_code/Curriculum-DPO/reward_models/llava-server
module load cuda/11.7
conda activate llava-server
echo $CONDA_DEFAULT_ENV
export CUDA_VISIBLE_DEVICES="0"
gunicorn "app:create_app()" &

sleep 40

#source /home/${USER}/.bashrc
conda activate dpo
cd /home/${USER}/public_code/Curriculum-DPO
echo $CONDA_DEFAULT_ENV
python reward_models/compute_text_align_score.py --save_path scores/lcm/finetuned_pickapic_text_align --dataset_path datasets_baseline/pickapic/test/lcm/llava_bertscore --split test