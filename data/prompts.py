
from datasets import load_dataset, load_from_disk
import os
import inflect
IE = inflect.engine()

def get_prompts(dataset, args):
    if dataset == 'pickapic':
        return get_pickapic_prompts(args.data_dir, args.subset)
    elif dataset == 'drawbench':
        return get_drawbench_prompts(args.data_dir, args.subset)
    else:
        if args.reward_fn =="aesthetic_score":
            return all_nouns()
        return all_nouns_activities()

def get_pickapic_prompts(data_dir, subset='test'):
    dataset = load_dataset("yuvalkirstain/pickapic_v1", cache_dir=data_dir, keep_in_memory=False)
    all_prompts = list(set([prompt for prompt in dataset[subset]['caption']]))
    return all_prompts

def get_drawbench_prompts(data_dir, subset='test'):
    ds = load_dataset("shunk031/DrawBench", cache_dir=data_dir)
    all_prompts = [example['prompts'] for example in ds[subset]]
    return all_prompts

def all_nouns_activities(nouns_file='assets/simple_animals.txt', activities_file='assets/activities.txt'):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    list_all_prompts = []
    for noun in nouns:
        for activity in activities:
            list_all_prompts.append(f"{IE.a(noun)} {activity}")
    return list_all_prompts

def all_nouns(nouns_file='assets/simple_animals.txt'):
    nouns = _load_lines(nouns_file)
    list_all_prompts = []
    for noun in nouns:
        list_all_prompts.append(f"A photo of {IE.a(noun)}")
    return list_all_prompts
def _load_lines(path):
    """
    Load lines from a file.
    """
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]
