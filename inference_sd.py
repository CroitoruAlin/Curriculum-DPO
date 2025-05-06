from config.args_sd import get_config
from diffusers import StableDiffusionPipeline, DDIMScheduler, EulerDiscreteScheduler
import torch
import safetensors
if __name__ == "__main__":
    args = get_config()
    pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model, revision=args.revision, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    # disable safety checker
    pipeline.safety_checker = None
    if args.lora_ckpt is not None:
        print("Using lora")
        state_dict = safetensors.torch.load_file(args.lora_ckpt)
        pipeline.load_lora_weights(state_dict, adapter_name='dpo')
        pipeline.set_adapters(["dpo"])
        pipeline.fuse_lora()

    pipeline.set_progress_bar_config(
        position=1,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.eval()
    image = pipeline(prompt=[args.prompt], num_inference_steps=args.num_steps, output_type="pil", height=args.resolution, width=args.resolution).images[0]
    image.save("result.png")
