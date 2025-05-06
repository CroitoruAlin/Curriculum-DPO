from config.args_lcm import get_config
from diffusers import AutoPipelineForText2Image, LCMScheduler
from data_generators.generate_data_lcm import get_module_kohya_state_dict
import torch
import safetensors
if __name__ == "__main__":
    args = get_config()
    pipeline = AutoPipelineForText2Image.from_pretrained(args.pretrained_model, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    if args.lora_ckpt is not None:
        state_dict = safetensors.torch.load_file(args.lora_ckpt)
        state_dict = get_module_kohya_state_dict(state_dict, "lora_unet", torch.float32, adapter_name='dpo')

        pipeline.load_lora_weights(state_dict, adapter_name='dpo')
        pipeline.set_adapters(["dpo"])
        pipeline.fuse_lora()

    pipeline.set_progress_bar_config(
        position=1,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.eval()
    image = pipeline(prompt=[args.prompt], num_inference_steps=args.num_steps, output_type="pil", height=args.resolution, width=args.resolution).images[0]
    image.save("result.png")
