# python train_stablediffusion.py \
#     --run_name dpu_text_align_animals \
#     --pretrained_model stable-diffusion-v1-5/stable-diffusion-v1-5 \
#     --prompt_fn nouns_activities \
#     --reward_fn llava_bertscore \
#     --dataset implicit_reward \
#     --data_dir /mnt/home/fmi2/vladh/data/curr_dpo/datasets/animals/train/sd/llava_bertscore \
#     --subset train \
#     --max_train_steps 10000 \
#     --update_frequency 500 \
#     --learning_rate 1e-5 \
#     --train_batch_size 16


# python train_stablediffusion.py \
#     --run_name dpu_text_align_drawbench \
#     --pretrained_model stabilityai/stable-diffusion-2-1-base \
#     --prompt_fn nouns_activities \
#     --reward_fn llava_bertscore \
#     --dataset implicit_reward \
#     --data_dir /mnt/home/fmi2/vladh/data/curr_dpo/datasets/drawbench/train/sd/llava_bertscore \
#     --subset train \
#     --max_train_steps 10000 \
#     --update_frequency 500 \
#     --learning_rate 1e-5 \
#     --train_batch_size 16

python train_lcm.py \
    --run_name dpo_lcm_text_align_drawbench \
    --prompt_fn nouns_activities \
    --reward_fn llava_bertscore \
    --dataset implicit_reward \
    --data_dir /mnt/home/fmi2/vladh/data/curr_dpo/lcm/datasets_lcm/drawbench/train/lcm/llava_bertscore/ \
    --subset train \
    --max_train_steps 10000 \
    --update_frequency 500 \
    --learning_rate 3e-4 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --lora_rank 64 \
    --lora_alpha 64 \
    --beta 200