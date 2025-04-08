from config.args_sd import get_config
from data.pickapic_dataset import PickaPicForScoring
from rewards import aesthetic_score
from tqdm import tqdm
import numpy as np
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

import argparse
reward_fn = aesthetic_score()

def score_paired_ds(args):
    prompts = []
    dict_result = {"image_0_score":[], "image_1_score":[]}
    dataset = load_from_disk(args.dataset_path).with_format('torch')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)
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
    dict_result = {"image_score":[]}
    dataset = load_from_disk(args.dataset_path).with_format('torch')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)
    for  entry in tqdm(dataloader):
            images = entry['image']
            prompts = entry['prompt']
            scores, _ = reward_fn(images, prompts, [])
            for i, score in enumerate(scores):
                dict_result["score"].append(score)
            images = []
            prompts = []
    score_ds = Dataset.from_dict(dict_result)
    score_ds.save_to_disk()
        
    print("Average: ", np.mean(dict_result["score"]))


if __name__ =="__main__":
    parser = argparse.ArgumentParser(prog='CL-DPO')
    
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="scores")
    parser.add_argument("--dataset_path", type=str, default="datasets/drawbench/test/sd")
    args = parser.parse_args()
    score_ds(args)
    

