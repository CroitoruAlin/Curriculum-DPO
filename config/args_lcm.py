import argparse
import time

def get_config(argv=None):
    parser = argparse.ArgumentParser(prog='CL-DPO')
    
    # ===== Curriculum DPO parameters ===== 
    parser.add_argument("--no_chunks", type=int, default=6)
    parser.add_argument("--run_type", type=str, default="dpo")
    parser.add_argument("--update_frequency", type=int, default=500)
    
    # ===== DPO parameters =====
    parser.add_argument("--beta", type=int, default=200)

    # ===== Hyperparameters =====
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--center_crop", type=bool, default=True)
    parser.add_argument("--random_flip", type=bool, default=True)
    parser.add_argument("--random_crop", type=bool, default=False)
    parser.add_argument("--no_hflip", type=bool, default=True)
    parser.add_argument("--max_train_samples", type=int, default=1000000)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--prediction_type", type=str, default=None)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--num_validation_images", type=int, default=10)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.25)
    parser.add_argument("--enable_xformers_memory_efficient_attention", type=bool, default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--cast_teacher_unet", type=bool, default=True)
    parser.add_argument("--w_min", type=int, default=5)
    parser.add_argument("--w_max", type=int, default=15)
    # ===== Logging =====
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    parser.add_argument("--run_name", type=str, default="dpo")
    # top-level logging directory for checkpoint saving.
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--output_dir", type=str, default=None)
   
    # ===== Pretrained Model =====
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    parser.add_argument("--pretrained_model", type=str, default='SimianLuo/LCM_Dreamshaper_v7')
    parser.add_argument("--lora_ckpt", type=str, default=None)
    parser.add_argument("--unet_time_cond_proj_dim", type=int, default=256)
    # ===== Pretrained Teacher Model ====
    parser.add_argument("--pretrained_teacher_model", default = "lykon/dreamshaper-7", type=str)
    # ===== Sampling =====
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--num_ddim_timesteps', type=int, default=50)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--sample_batch_size", type=int, default=10)

     # ===== Training =====
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--huber_c",default = 0.001, type=float)
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--allow_tf32", type=bool, default=True)
    parser.add_argument("--lora_rank", type=int, default=8)
    # batch size (per GPU!) to use for training.
    parser.add_argument("--train_batch_size", type=int, default=8)
    # learning rate.
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default='constant')
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--cfg", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=None)

    #===== Prompt and Reward Function =====
    # prompt function to use. see `prompts.py` for available prompt functions.
    parser.add_argument("--prompt_fn", type=str, default="nouns_activities")
    parser.add_argument("--reward_fn", type=str, default="llava_bertscore")
    parser.add_argument("--prompt", type=str, default="A red car")
    
    #===== Data set name =====
    parser.add_argument("--dataset", type=str, default="drawbench", choices=['pickapic', 'drawbench', 'animals'],)
    parser.add_argument("--subset", type=str, default="test")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--score_dir", type=str, default=None)
    parser.add_argument("--no_generated_images_per_prompt", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="datasets")

    if argv is not None:
        args = parser.parse_args(argv[1:])
    else:
        args = parser.parse_args()
    if args.subset=='train':
        args.no_generated_images_per_prompt=500
    else:
        args.no_generated_images_per_prompt=10
    args.tracker_project_name = args.run_name
    args.output_dir = f"experiments/{args.run_name}"
    return args
    