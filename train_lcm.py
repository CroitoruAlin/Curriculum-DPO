import argparse
import functools
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
import time
from typing import List, Union

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from braceexpand import braceexpand
from huggingface_hub import create_repo
from packaging import version
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import default_collate, Dataset
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig

from utils.utils_train import process_single_batch
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    LCMScheduler, AutoPipelineForText2Image,
    DiffusionPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from data.dataset_factory import get_dataset
from data.utils import Preprocess
from config.args_lcm import get_config
import safetensors

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__)


def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(module, adapter_name=adapter_name).items():
        # print(peft_key)
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(module.peft_config[adapter_name].lora_alpha).to(dtype)

    return kohya_ss_state_dict



def log_validation(vae, unet, validation_prompts, args, accelerator, weight_dtype, step, ref=False, seed=None):
    logger.info("Running validation... ")

    unet = accelerator.unwrap_model(unet)
    
    
    pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model, torch_dtype=torch.float16, safety_checker=None)
    lora_state_dict = get_module_kohya_state_dict(unet, "lora_unet", torch.float16, adapter_name='dpo')
    
    if not ref:
        pipeline.load_lora_weights(lora_state_dict, adapter_name='dpo')
        pipeline.set_adapters(["dpo"])
        pipeline.fuse_lora()
    pipeline.unet = pipeline.unet.to(torch.float16)
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device, dtype=torch.float16)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    
    image_logs = []

    for _, prompt in enumerate(validation_prompts):
        images = []
        # with torch.autocast("cuda", dtype=torch.float16):
        images = pipeline(
                prompt=prompt,
                num_inference_steps=8,
                num_images_per_prompt=1,
                generator=generator,
                guidance_scale=8.0,
                lcm_origin_steps=50
            ).images
        image_logs.append({"validation_prompt": prompt, "images": images})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                for image in images:
                    # print(image.size)
                    image = wandb.Image(image, caption=validation_prompt if not ref else validation_prompt+"ref")
                    formatted_images.append(image)

            tracker.log({"validation" if not ref else "validation_ref": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


# From LatentConsistencyModel.get_guidance_scale_embedding
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision, use_auth_token=True
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")



# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds


def main(args):
    logging_dir = Path(args.output_dir, args.logdir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 1. Create the noise scheduler and the desired noise schedule.
    # print("Args:",args.items())
    # print("Pretrained:", args.pretrained.items())
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_teacher_model, subfolder="scheduler", revision=None
    )

    # The scheduler calculates the alpha and sigma schedule for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model, subfolder="tokenizer", revision=None, use_fast=False
    )

    # 3. Load text encoders
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="text_encoder", revision=None
    )

    # 4. Load VAE from SD-XL checkpoint (or more stable VAE)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
        revision=None
    )

    # 5. Load teacher U-Net
    teacher_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="unet", torch_dtype=torch.float16, variant="fp16", revision=None
    )

    # 6. Freeze teacher vae, text_encoder, and teacher_unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)

    # 7. Create online (`unet`) student U-Nets.
    time_cond_proj_dim = (
        teacher_unet.config.time_cond_proj_dim
        if teacher_unet.config.time_cond_proj_dim is not None
        else args.unet_time_cond_proj_dim
    )
    args.unet_time_cond_proj_dim = time_cond_proj_dim
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder='unet')

    target_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder='unet')

    unet_ref = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder='unet')
    unet_ref.train()
    unet_ref.requires_grad_(False)
    target_unet.train()
    target_unet.requires_grad_(False)
    
    unet.train()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # 8. Add LoRA to the student U-Net, only the LoRA projection matrix will be updated by the optimizer.
    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ],
    )
    unet = get_peft_model(unet, lora_config, adapter_name='dpo')
 
    unet.set_adapters(["dpo"])

    # print([name for name, _ in unet.named_parameters()])
    for name, param in unet.named_parameters():
        if 'dpo' in name:
            param.requires_grad_(True)
    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Move teacher_unet to device, optionally cast to weight_dtype
    teacher_unet.to(accelerator.device)
    if args.cast_teacher_unet:
        teacher_unet.to(dtype=weight_dtype)

    unet_ref.to(accelerator.device)
    target_unet.to(accelerator.device)
    
    if args.cast_teacher_unet:
        unet_ref.to(dtype=weight_dtype)
        target_unet.to(dtype=weight_dtype)
    

    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    solver = solver.to(accelerator.device)

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unet_ = accelerator.unwrap_model(unet)
                lora_state_dict = get_peft_model_state_dict(unet_, adapter_name="dpo")
                StableDiffusionPipeline.save_lora_weights(os.path.join(output_dir, "unet_lora"), lora_state_dict)
                # save weights in peft format to be able to load them back
                unet_.save_pretrained(output_dir)

                for _, model in enumerate(models):
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            # load the LoRA into the model
            unet_ = accelerator.unwrap_model(unet)
            unet_.load_adapter(input_dir, "default", is_trainable=True)

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 11. Enable optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
            unet_ref.enable_xformers_memory_efficient_attention()
            target_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer_class = torch.optim.AdamW

    # 12. Optimizer creation
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
        return {"prompt_embeds": prompt_embeds}

    train_dataset = get_dataset(args)
    preprocessor = Preprocess(tokenizer, args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=preprocessor.collate_fn,
        batch_size=args.train_batch_size,
        num_workers=7, drop_last=True
    )
    train_dataloader.num_batches = len(train_dataset) // args.train_batch_size
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.run_name, config=tracker_config)

    uncond_input_ids = tokenizer(
        [""] * args.train_batch_size, return_tensors="pt", padding="max_length", max_length=77
    ).input_ids.to(accelerator.device)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {train_dataloader.num_batches}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Steps 20.4.4 - 20.4.12
                model_pred_w, target_pred_w, target_ref_w, model_ref_pred_w, model_ref_pred_l, model_pred_l, target_pred_l, target_ref_l, index = process_single_batch(batch, teacher_unet, unet, unet_ref, unet_ref, accelerator, vae,
                                                                                      compute_embeddings_fn, noise_scheduler, solver,
                                                                                      scalings_for_boundary_conditions, predicted_origin, weight_dtype, args,
                                                                                      append_dims, alpha_schedule, sigma_schedule,
                                                                                      uncond_prompt_embeds)
                # 20.4.13. Calculate loss
                
                loss_ref_w = torch.sqrt((model_ref_pred_w.float() - target_ref_w.float()) ** 2 + args.huber_c**2) - args.huber_c
                loss_ref_l = torch.sqrt((model_ref_pred_l.float() - target_ref_l.float()) ** 2 + args.huber_c**2) - args.huber_c
                
                loss_pred_w = torch.sqrt((model_pred_w.float() - target_pred_w.float()) ** 2 + args.huber_c**2) - args.huber_c
                loss_pred_l = torch.sqrt((model_pred_l.float() - target_pred_l.float()) ** 2 + args.huber_c**2) - args.huber_c
                
                loss_pred_ref_w = torch.sqrt((model_pred_w.float() - target_ref_w.float()) ** 2 + args.huber_c**2) - args.huber_c
                loss_pred_ref_l = torch.sqrt((model_pred_l.float() - target_ref_l.float()) ** 2 + args.huber_c**2) - args.huber_c
                loss_pred_w = torch.mean(loss_pred_w, dim=(1,2,3))
                loss_ref_w = torch.mean(loss_ref_w, dim=(1,2,3))
                loss_pred_l = torch.mean(loss_pred_l, dim=(1,2,3))
                loss_ref_l = torch.mean(loss_ref_l, dim=(1,2,3))
                loss_pred_ref_w = torch.mean(loss_pred_ref_w, dim=(1,2,3))
                loss_pred_ref_l = torch.mean(loss_pred_ref_l, dim=(1,2,3))
                    
                # loss = torch.mean(loss_pred_w) + torch.mean(loss_pred_l)
                start_timesteps = solver.ddim_timesteps[index]
                weight = args.beta#*start_timesteps
                loss = -torch.mean(torch.log(torch.sigmoid(-weight*((loss_pred_w - loss_ref_w) - (loss_pred_l - loss_ref_l)))))
 
                
                # 20.4.14. Backpropagate on the online student model (`unet`)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0 or global_step in [1, 50, 100, 150, 200, 250]:
                        validation_prompts = random.sample(train_dataset.usable_prompts, 5)
                        seed = random.randint(0, 50000)
                        log_validation(vae, unet, validation_prompts, args, accelerator, weight_dtype, global_step, seed=seed)
                        log_validation(vae, unet, validation_prompts, args, accelerator, weight_dtype, global_step, ref=True, seed=seed)
                   
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "diff_w:": torch.mean(loss_pred_w - loss_ref_w),
                "diff_l": torch.mean(loss_pred_l - loss_ref_l)}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step % args.update_frequency==0:
                train_dataloader.dataset.update_cl()
                train_dataloader.num_batches = len(train_dataset) // args.train_batch_size
                num_update_steps_per_epoch = math.ceil((train_dataloader.num_batches) / (args.gradient_accumulation_steps))
                logger.info(f"Length train_dataloader:{len(train_dataloader)}, num_update_steps_per_epoch: {num_update_steps_per_epoch}, epoch:{epoch}")
                if overrode_max_train_steps:
                    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
                progress_bar = tqdm(
                range(0, args.max_train_steps),
                initial=global_step,
                desc="Steps",
                # Only show the progress bar once on each machine.
                disable=not accelerator.is_local_main_process,
                        )
            if global_step >= args.max_train_steps:
                break
        if epoch%1==0:#(args.max_train_steps <= global_step and global_step <args.args.update_frequency):
            train_dataloader.dataset.update_cl()
            train_dataloader.num_batches = len(train_dataset) // args.train_batch_size
            num_update_steps_per_epoch = math.ceil((train_dataloader.num_batches) / (args.gradient_accumulation_steps))
            logger.info(f"Length train_dataloader:{len(train_dataloader)}, num_update_steps_per_epoch: {num_update_steps_per_epoch}, epoch:{epoch}")
            if overrode_max_train_steps:
                args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
            progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
                        )
                
    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        lora_state_dict = get_peft_model_state_dict(unet, adapter_name="dpo")
        StableDiffusionPipeline.save_lora_weights(os.path.join(args.output_dir, "unet_lora"), lora_state_dict)

    accelerator.end_training()

def debug_function(vae, pred_w, pred_l, name_w, name_l):
    reconstructed_pred_w = vae.decode((pred_w[:1]/vae.config.scaling_factor).to(dtype=torch.float16)).sample
    recontruscted_ref_w = vae.decode((pred_l[:1]/vae.config.scaling_factor).to(dtype=torch.float16)).sample
                
    # print("----Debug----")
    # print(torch.amax(reconstructed_pred_w))
    # print(torch.amin(reconstructed_pred_w))

    image1 = reconstructed_pred_w[0]
    image1 *=127.5
    image1 +=127.5
    image1 = image1.cpu().detach().numpy()
    image1 = image1.swapaxes(0, -1).swapaxes(0,1)
    Image.fromarray(image1.astype(np.uint8)).save(name_w)
    image2 = recontruscted_ref_w[0]
    image2 *=127.5
    image2 +=127.5
    image2 = image2.cpu().detach().numpy()
    image2 = image2.swapaxes(0, -1).swapaxes(0,1)
    Image.fromarray(image2.astype(np.uint8)).save(name_l)
if __name__ == "__main__":
    config = get_config()
    main(config)