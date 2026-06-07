import sys
import os

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)
from rewards import human_score
from tqdm import tqdm
import numpy as np
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import io
import torch
reward_fn, preprocess_val, tokenizer = human_score()
resolution = 512
def collate_fn_pairs(entries):
    new_images_0 = []
    new_images_1 = []
    new_prompts = []
    text_prompts = []
    for entry in entries:
        image_0 = entry.get("jpg_0", entry['image_0'])
        image_1 = entry.get("jpg_1", entry['image_1'])
        prompts = entry.get("caption", entry['prompt'])
        if not isinstance(image, Image.Image):
            new_images_0.append(preprocess_val(Image.open(io.BytesIO(image)).resize((resolution, resolution))))
            new_images_1.append(preprocess_val(Image.open(io.BytesIO(image_1)).resize((resolution, resolution))))
        else:
            new_images_0.append(preprocess_val(image.resize((resolution, resolution))))
            new_images_1.append(preprocess_val(image_1.resize((resolution, resolution))))
        new_prompts.append(tokenizer(prompts))
        text_prompts.extend(prompts)
    # print(text_prompts)
    return {"image_0": torch.stack(new_images_0, dim=0), "image_1": torch.stack(new_images_1, dim=0), "prompt": torch.cat(new_prompts, dim=0), "text_prompt":text_prompts}

def collate_fn(entries):
    new_images = []
    new_prompts = []
    text_prompts = []
    for entry in entries:
        image = entry['image']
        prompts = entry.get("caption", entry['prompt'])
        if not isinstance(image, Image.Image):
            new_images.append(preprocess_val(Image.open(io.BytesIO(image)).resize((resolution, resolution))))
        else:
            new_images.append(preprocess_val(image.resize((resolution, resolution))))
        new_prompts.append(tokenizer(prompts))
        # print(prompts)
        text_prompts.append(prompts)
    return {"image": torch.stack(new_images, dim=0), "prompt": torch.cat(new_prompts, dim=0),  "text_prompt":text_prompts}

def score_paired_ds(args):
    prompts = []
    dict_result = {"image_0_score":[], "image_1_score":[]}
    dataset = load_from_disk(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=collate_fn_pairs)
    for  entry in tqdm(dataloader):
            image_0 = entry.get("jpg_0", entry['image_0']) 
            image_1 = entry.get("jpg_1", entry['image_1']) 
            prompts = entry.get("caption", entry['prompt'])
            scores_0, _ = reward_fn(image_0, prompts, [])
            scores_1, _ = reward_fn(image_1, prompts, [])
            for i, score in enumerate(scores_0):
                score_0 = score
                score_1 = scores_1[i]
                dict_result["image_0_score"].append(score_0)
                dict_result["image_1_score"].append(score_1)
            images = []
            prompts = []
    score_ds = Dataset.from_dict(dict_result)
    score_ds.save_to_disk()
        
    print("Average: ", 0.5*(np.mean(dict_result["image_0_score"]) + np.mean(dict_result["image_1_score"])))

def score_ds(args):
    prompts = []
    dict_result = {"score":[]}
    dataset = load_from_disk(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn)
    for  entry in tqdm(dataloader):
            images = entry['image']
            prompts = entry['prompt']
            scores = reward_fn(images.to("cuda"), prompts.to("cuda"), [])
            for i, score in enumerate(scores):
                dict_result["score"].append(score)
            images = []
            prompts = []
    score_ds = Dataset.from_dict(dict_result)
    os.makedirs(args.save_path, exist_ok=True)
    score_ds.save_to_disk(args.save_path)
        
    print("Average: ", np.mean(dict_result["score"]))

def score_ds_head2head_comparison(args):
    prompts = []
    dict_result = {"score":[]}
    dataset_paths = args.dataset_path.split(",")
    dict_result = {}

    for dataset_path in dataset_paths:
        dataset = load_from_disk(dataset_path)
        dataset = dataset.sort("prompt")
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn)
        dict_result[dataset_path] = {}
        for  entry in tqdm(dataloader):
                images = entry['image']
                prompts = entry['prompt']
                text_prompts = entry['text_prompt']
                scores = reward_fn(images.to("cuda"), prompts.to("cuda"), [])
                # print(text_prompts)
                for i, score in enumerate(scores):
                    prompt = text_prompts[i]
                    if prompt in dict_result[dataset_path]:
                        dict_result[dataset_path][prompt].append(score)
                    else:
                        dict_result[dataset_path][prompt] = [score]
    ds_path = dataset_paths[0]
    prompts = list(dict_result[ds_path].keys())
    
    final_results = np.zeros((len(dataset_paths), 10*len(prompts)))
    dict_final_results = {}
    for prompt in prompts:

        for ds_path in dataset_paths:
            if ds_path not in dict_final_results:
                dict_final_results[ds_path] = dict_result[ds_path][prompt][:10]
            else:
                dict_final_results[ds_path].extend(dict_result[ds_path][prompt][:10])
    print(final_results.shape, np.array(dict_final_results[ds_path]).shape)
    for i,ds_path in enumerate(dict_final_results):
        final_results[i] = np.array(dict_final_results[ds_path])
    
    final_results =  final_results.T
    predictions = np.zeros(final_results.shape)
    for i in range(final_results.shape[0]):
        predictions[i, np.argmax(final_results[i])] = 1
    print(predictions.sum(axis=0)/predictions.sum())

    print(final_results.shape)
if __name__ =="__main__":
    parser = argparse.ArgumentParser(prog='CL-DPO')
    
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="scores/human")
    parser.add_argument("--dataset_path", type=str, default="datasets/drawbench/test/sd")
    parser.add_argument("--split", type=str, default='train')
    args = parser.parse_args()
    if ',' in args.dataset_path:
        score_ds_head2head_comparison(args)
    elif 'pickapic' in args.dataset_path and args.split=='train':
        score_paired_ds(args)
    else:
        score_ds(args)
    

