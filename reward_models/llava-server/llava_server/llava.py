from typing import Iterable, List
from typing import Iterable, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from llava_local.utils import disable_torch_init
from llava_local.conversation import conv_templates, SeparatorStyle
from llava_local.model.builder import load_pretrained_model
from llava_local.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava_local.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import CLIPImageProcessor
from PIL import Image
#from llava.conversation import simple_conv_multimodal
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

MAX_TOKENS = 64

PROMPT = "Human:"
def load_phi3():


    model_id = "microsoft/Phi-3-vision-128k-instruct" 

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 
    @torch.inference_mode()
    def inference_fn(
        images: Iterable[Image.Image], queries: Iterable[Iterable[str]]
    ) -> List[List[str]]:
        messages = [ {"role": "user", "content": f"<|image_{i+1}|>\n{query[0]}"} for i, query in enumerate(queries) ] 

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

        generation_args = { 
            "max_new_tokens": 500, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        return response
    return inference_fn

def load_llava(params_path):
    # load model
    disable_torch_init()
    model_path="llava-v1.5-13b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    # print(list([name for name,_ in model.named_parameters()]))
    model = model.cuda()

    # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    # # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    # # if mm_use_im_start_end:
    # #     tokenizer.add_tokens(
    # #         [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    # #     )

    # vision_tower = model.model.vision_tower#[0]
    # vision_tower.to(device="cuda", dtype=torch.float16)
    # vision_config = vision_tower.config
    # vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
    #     [DEFAULT_IMAGE_PATCH_TOKEN]
    # )[0]
    # vision_config.use_im_start_end = mm_use_im_start_end
    # if mm_use_im_start_end:
    #     (
    #         vision_config.im_start_token,
    #         vision_config.im_end_token,
    #     ) = tokenizer.convert_tokens_to_ids(
    #         [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
    #     )
    # image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    # if mm_use_im_start_end:
    #     image_tokens = (
    #         DEFAULT_IM_START_TOKEN
    #         + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    #         + DEFAULT_IM_END_TOKEN
    #     )
    # else:
    #     image_tokens = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"

    @torch.inference_mode()
    def inference_fn(
        images: Iterable[Image.Image], queries: Iterable[Iterable[str]]
    ) -> List[List[str]]:
        assert len(images) == len(queries)
        assert np.all(len(queries[0]) == len(q) for q in queries)
        # print(queries)
        # queries = np.array(queries)  # (batch_size, num_queries_per_image)

        # preprocess images
        images = image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        images = images.to("cuda", dtype=torch.float16)

        # first, get the activations for the image tokens
        #initial_prompts = [PROMPT + image_tokens + " " for _ in range(len(images))]
        
        queries_input_ids=  []
        for query in queries:
            qs = query[0]
            for i, q in enumerate(query):
                if i>0:
                    qs=qs+q
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = conv_templates['llava_v1'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            
            prompt = conv.get_prompt()
            # print(prompt)
            query_input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            queries_input_ids.append(query_input_ids)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        input_ids = torch.cat(tuple(queries_input_ids), dim=0)
        output_ids = model.generate(
                input_ids.cuda(),
                images=images.cuda(),
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)
        
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        outputs = [output.strip() for output in outputs]
        for i, output in enumerate(outputs):
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            outputs[i] = output.strip()
        # reshape outputs back to (batch_size, num_queries_per_image)
        outputs_clean = np.array(outputs).reshape(np.array(queries).shape)

        return outputs_clean.tolist()

    return inference_fn
