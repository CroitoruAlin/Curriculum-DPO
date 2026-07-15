import sys
import os
sys.path.insert(0, os.getcwd())
from absl import app, flags
from config.args_sd import get_config
from data.prompts import get_prompts
from ml_collections import config_flags
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
import numpy as np
import torch
from functools import partial
import tqdm
from PIL import Image
from datasets import load_dataset, Features, Image, Value, Dataset
import safetensors
from peft import get_peft_model, PeftModel
from diffusers.utils import convert_state_dict_to_diffusers
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

def main():
    config = get_config()

    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained_model, revision=config.revision, torch_dtype=torch.float16)
    # pipeline = pipeline.to("cuda")
    # disable safety checker
    pipeline.safety_checker = None
    if config.lora_ckpt is not None:
        for i, lora_ckpt in enumerate(config.lora_ckpt.split(",")):
            state_dict = safetensors.torch.load_file(lora_ckpt)
            new_state = {}
            for key in state_dict:
                new_state[key.replace("unet.base_model.model", "unet")] = state_dict[key]
            pipeline.load_lora_weights(new_state, adapter_name=f'dpo_{i}')
            pipeline.set_adapters([f'dpo_{i}'])
            pipeline.fuse_lora()
            pipeline.unload_lora_weights()
    if config.save_unet_path is not None:
        os.makedirs(config.save_unet_path, exist_ok=True)
        pipeline.unet.save_pretrained(config.save_unet_path)
        print(f"Fused UNet successfully saved to {config.save_unet_path}")

    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # prepare prompts
    all_prompts = get_prompts(config.dataset, config)
    print(f"Number of prompts: {len(all_prompts)}")
    batch_size=config.sample_batch_size
    pipeline = pipeline.to("cuda")
    pipeline.unet.eval()
    generator = torch.Generator("cuda").manual_seed(123456)
    def generate_images():
        for i in tqdm(range(0, config.no_generated_images_per_prompt)):
            for start_index in tqdm(range(0, len(all_prompts), batch_size)):
                #################### SAMPLING ####################
                prompts = all_prompts[start_index:start_index+batch_size]
                images = pipeline(prompt=prompts, num_inference_steps=config.num_steps, output_type="pil", height=config.resolution, width=config.resolution, generator=generator).images
                for i, image in enumerate(images):
                    if not os.path.exists("result.png"):
                        image.save("result.png")
                        # exit()
                    yield {"image": image, "prompt": prompts[i]}
    features = Features({"image": Image(),
                         "prompt": Value("string")})
    dataset = Dataset.from_generator(generator = generate_images,
                                     features = features)
    dataset_path = f"{config.save_path}/{config.dataset}/{config.subset}/sd2.1_dpo_baseline/{config.reward_fn}"
    os.makedirs(dataset_path, exist_ok=True)
    dataset.save_to_disk(dataset_path)


if __name__ == "__main__":
    main()