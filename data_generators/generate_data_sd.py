import sys
import os
sys.path.insert(0, os.getcwd())
from absl import app, flags
from config.args_sd import get_config
from data.prompts import get_prompts
from ml_collections import config_flags
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerDiscreteScheduler
import numpy as np
import torch
from functools import partial
import tqdm
from PIL import Image
from datasets import load_dataset, Features, Image, Value, Dataset
import safetensors


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

def main():
    config = get_config()

    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained_model, revision=config.revision, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    # disable safety checker
    pipeline.safety_checker = None
    if config.lora_ckpt is not None:
        state_dict = safetensors.torch.load_file(config.lora_ckpt)
        pipeline.load_lora_weights(state_dict, adapter_name='dpo')
        pipeline.set_adapters(["dpo"])
        pipeline.fuse_lora()

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
    if config.generate_negative_samples:
        all_prompts, all_negative_prompts = get_prompts(config.dataset, config)
    else:
        all_prompts = get_prompts(config.dataset, config)
        print(f"Number of prompts: {len(all_prompts)}")
    batch_size=config.sample_batch_size

    pipeline.unet.eval()
    def generate_images():
        for i in tqdm(range(0, config.no_generated_images_per_prompt)):
            for start_index in tqdm(range(0, len(all_prompts), batch_size)):
                #################### SAMPLING ####################
                prompts = all_prompts[start_index:start_index+batch_size]
                images = pipeline(prompt=prompts, num_inference_steps=config.num_steps, output_type="pil", height=config.resolution, width=config.resolution).images
                for i, image in enumerate(images):
                    yield {"image": image, "prompt": prompts[i], "degradation": "None", "mask_ratio": 0.0}
                    
                # Mask input text embedding:
                if False:
                    for mask_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        images = pipeline(prompt=prompts, num_inference_steps=config.num_steps, mask_ratio=mask_ratio, output_type="pil", height=config.resolution, width=config.resolution).images
                        for i, image in enumerate(images):
                            yield {"image": image, "prompt": prompts[i], "degradation": "None", "mask_ratio": mask_ratio}
                    
                if config.generate_negative_samples:
                    images = pipeline(prompt=prompts, num_inference_steps=config.num_steps // 2, output_type="pil", height=config.resolution, width=config.resolution).images
                    for i, image in enumerate(images):
                        yield {"image": image, "prompt": prompts[i], "degradation": "half_steps"}
                    
            if config.generate_negative_samples:
                for degradation, negative_prompts in all_negative_prompts.items():
                    for start_index in tqdm(range(0, len(negative_prompts), batch_size)):
                        #################### SAMPLING ####################
                        prompts = negative_prompts[start_index:start_index+batch_size]
                        images = pipeline(prompt=prompts, num_inference_steps=config.num_steps, output_type="pil", height=config.resolution, width=config.resolution).images
                        for i, image in enumerate(images):
                            yield {"image": image, "prompt": prompts[i], "degradation": degradation}

    features = Features(
        {
            "image": Image(),
            "prompt": Value("string"),
            "degradation": Value("string"),
            "mask_ratio": Value("float16")
        }
    )
    dataset = Dataset.from_generator(generator = generate_images,
                                     features = features)
    dataset_path = f"{config.save_path}/{config.dataset}/{config.subset}/sd/{config.reward_fn}"
    os.makedirs(dataset_path, exist_ok=True)
    dataset.save_to_disk(dataset_path)


if __name__ == "__main__":
    main()
