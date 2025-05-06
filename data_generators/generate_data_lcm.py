import sys
import os
sys.path.insert(0, os.getcwd())
from absl import app, flags
from config.args_lcm import get_config
from data.prompts import get_prompts
from ml_collections import config_flags
from accelerate.logging import get_logger
from diffusers import AutoPipelineForText2Image, LCMScheduler, EulerDiscreteScheduler
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

    pipeline = AutoPipelineForText2Image.from_pretrained(config.pretrained_model, safety_checker=None, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    # disable safety checker
    pipeline.safety_checker = None
    if config.lora_ckpt is not None:
        state_dict = safetensors.torch.load_file(config.lora_ckpt)
        state_dict = get_module_kohya_state_dict(state_dict, "lora_unet", torch.float32, adapter_name='dpo')

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
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

    # prepare prompts
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
                    yield {"image": image, "prompt": prompts[i]}
    features = Features({"image": Image(),
                         "prompt": Value("string")})
    dataset = Dataset.from_generator(generator = generate_images,
                                     features = features)
    dataset_path = f"{config.save_path}/{config.dataset}/{config.subset}/lcm/{config.reward_fn}"
    os.makedirs(dataset_path, exist_ok=True)
    dataset.save_to_disk(dataset_path)

def get_module_kohya_state_dict(peft_dict, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in peft_dict.items():
        # print(peft_key)
        kohya_key = peft_key.replace("unet.base_model.model", prefix)
        kohya_key = kohya_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(8).to(dtype)

    return kohya_ss_state_dict

if __name__ == "__main__":
    main()
