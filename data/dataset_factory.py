import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from PIL import Image
import io
from tqdm import tqdm
import datasets

def decide_threshold(reward_type):
    if reward_type=='llava_bertscore':
        return 0.02, 0.02
    elif reward_type=="human_score":
        return 0.02, 0.18
    elif reward_type=="aesthetic_score":
        return 0.4, 4.
    else:
        return 0.1
    

class EntryInfo:

    def __init__(self, index, score, prompt):
        self.index = index
        self.score = score
        self.prompt = prompt

class PromptsDataset(Dataset):
    def __init__(self, logger, data_dir="datasets/drawbench/test/sd", score_dir="scores/pickapic/human",
                 groups=5, split='train', reward='aesthetic_score', ):
        self.paths = []
        self.prompts = []
        logger.info("Start reading data")
        self.dataset = load_from_disk(data_dir)
        logger.info("Data read")
        logger.info("Start reading scores")
        self.scores = load_from_disk(score_dir)
        logger.info("Scores read")
        score_images = np.array([score for score in self.scores['score']])
        if split is not None:
            self.dataset = self.dataset[split]
        self.dataset = self.dataset.add_column(name="score", column = [score for score in self.scores['score']])
        keep_threshold, threshold_per_sample = decide_threshold(reward)
        scores_filtering = np.array(self.dataset["score"])
        indices = np.where(scores_filtering > threshold_per_sample)[0]
        self.dataset = self.dataset.select(indices)
        prompts_to_indices = {}
        new_dataset = datasets.Dataset.from_dict({
                "prompt": self.dataset["prompt"],
                "score": self.dataset['score']
            })
        
        for i, entry in enumerate(tqdm(new_dataset)):
            if entry['prompt'] not in prompts_to_indices:
                prompts_to_indices[entry['prompt']] = [EntryInfo(i, entry['score'], entry['prompt'])]
            else:
                prompts_to_indices[entry['prompt']].append(EntryInfo(i, entry['score'], entry['prompt']))
        self.pairs_per_prompt = {}
        self.thresholds_diff = {}
        self.index_curr = {}
        self.usable_pairs = {}
        logger.info("Start creating pairs")
        for prompt in tqdm(prompts_to_indices):
            entries = prompts_to_indices[prompt]
            # subset1 = random.sample(entries, 100)
            # subset2 = random.sample(entries, 100)
            max_diff = 0
            min_diff = 1000
            self.pairs_per_prompt[prompt] = []
            for i, first_entry in enumerate(entries[:-1]):
                for second_entry in entries[i+1:]:
            # for first_entry in subset1:
            #     for second_entry in subset2:
                    if abs(first_entry.score-second_entry.score)>keep_threshold:
                        self.pairs_per_prompt[prompt].append((first_entry.index, second_entry.index, abs(first_entry.score-second_entry.score)))
                        if max_diff < abs(first_entry.score-second_entry.score):
                            max_diff = abs(first_entry.score-second_entry.score)
                        if min_diff > abs(first_entry.score-second_entry.score):
                            min_diff = abs(first_entry.score-second_entry.score)
            self.thresholds_diff[prompt] = np.linspace(min_diff, max_diff, groups)[::-1]
            self.index_curr[prompt] = 0
            self.usable_pairs[prompt] = self.get_usable_pairs(prompt)
        self.transform = None
        self.usable_prompts = list(self.thresholds_diff.keys())
        self.all_usable_pairs = []
        self.all_possible_pairs = []
        for prompt in self.usable_prompts:
            self.all_usable_pairs.extend(self.usable_pairs[prompt])
            self.all_possible_pairs.extend(self.pairs_per_prompt[prompt])
        logger.info(f"Length total pairs = {len(self.all_possible_pairs)}")
        logger.info(f"Length selected pairs = {len(self.all_usable_pairs)}")
    
    def get_usable_pairs(self, prompt):
        usable_pairs = []
        unique_indices = []
        if len(self.thresholds_diff[prompt])<=self.index_curr[prompt]+1:
            return self.use_entire_ds(prompt)
        for index_1,index_2,score_diff in self.pairs_per_prompt[prompt]:
            if score_diff >=self.thresholds_diff[prompt][self.index_curr[prompt]+1]:
                usable_pairs.append((index_1, index_2, score_diff))
                unique_indices.append(index_1)
        return usable_pairs

    def update_cl(self):
        for prompt in self.thresholds_diff:
            if self.index_curr[prompt] < len(self.thresholds_diff[prompt])-2:
                self.index_curr[prompt]+=1
                self.usable_pairs[prompt] = self.get_usable_pairs(prompt)
        self.all_usable_pairs = []
        for prompt in self.usable_pairs:
            self.all_usable_pairs.extend(self.usable_pairs[prompt])
        print(f"Length total pairs = {len(self.all_possible_pairs)}")
        print(f"Length selected pairs = {len(self.all_usable_pairs)}")
    
    def use_entire_ds(self, prompt):
        self.index_curr[prompt] = len(self.thresholds_diff)-2
        return self.pairs_per_prompt[prompt]

    def __len__(self):
        return len(self.all_usable_pairs)

    def __getitem_without_transform(self, ind):
        index_1, index_2, _ = self.all_usable_pairs[ind]
        score1 = self.dataset[index_1]['score']
        prompt = self.dataset[index_1]['prompt']
        score2 = self.dataset[index_2]['score']
        image1 = self.dataset[index_1]['image']
        image2 = self.dataset[index_2]['image']
        
        if score2 > score1:
            return image2, image1, " ".join(prompt.split("_"))
        else:
            return image1, image2, " ".join(prompt.split("_"))
    def __getitem__(self, ind):
        image_w, image_l, prompt = self.__getitem_without_transform(ind)
        if self.transform is not None:
            return self.transform(image_w, image_l, prompt)
        return image_w, image_l, prompt
       
    def set_transform(self, transform):
        self.transform = transform


class PickaPic(Dataset):

    def __init__(self, data_dir='datasets/pickapic_v1', score_dir="scores/pickapic/human",
                 split="train", groups = 5, reward='aesthetic_score'):
        if 'llava' in reward:
            self.dataset = load_from_disk("/home/fl488644/Curriculum-DPO/datasets/pickapic_150k", keep_in_memory=False)#
        else:
            self.dataset = load_dataset(data_dir,cache_dir=None, keep_in_memory=False, split=split)#load_from_disk("/home/fl488644/Curriculum-DPO/datasets/pickapic_150k", keep_in_memory=False)##load_from_disk("/home/fl488644/Curriculum-DPO/datasets/pickapic_150k", keep_in_memory=False)#load_dataset(data_dir,cache_dir=None, keep_in_memory=False)
        self.scores = load_from_disk(score_dir, keep_in_memory=True)
        scores_images_0 = np.array([score for score in self.scores['image_0_score']])
        scores_images_1 = np.array([score for score in self.scores['image_1_score']])
        orig_len = self.dataset.num_rows
        self.original_dataset = self.dataset#[split]
        # print(len(self.original_dataset), len(self.scores))
        if 'llava' not in reward:
            self.original_dataset = self.original_dataset.add_column(name="image_0_score", column = [score for score in self.scores['image_0_score']])
            self.original_dataset = self.original_dataset.add_column(name="image_1_score", column = [score for score in self.scores['image_1_score']])
        not_split_idx = []
        print(scores_images_0.shape, scores_images_1.shape, self.original_dataset['label_0'][0])
        keep_threshold, _ = decide_threshold(reward)
        for i, label_0 in tqdm(enumerate(self.original_dataset['label_0'])):
            if label_0 in (0,1) and abs(scores_images_0[i] - scores_images_1[i])>keep_threshold and(scores_images_0[i]>0.1 and scores_images_1[i]>0.1):
                not_split_idx.append(i)
        self.used_dataset = self.original_dataset.select(not_split_idx)
        scores_images_0 = scores_images_0[not_split_idx]
        scores_images_1 = scores_images_1[not_split_idx]
        new_len =   self.used_dataset.num_rows
        print(f"Eliminated {orig_len - new_len}/{orig_len} split decisions for Pick-a-pic")
   
        self.num_used_entries =   self.used_dataset.num_rows

        self.indices_diff_scores = []
        for i in tqdm(range(self.num_used_entries)):
            score_0 = scores_images_0[i]
            score_1 = scores_images_1[i]
            self.indices_diff_scores.append([i, abs(score_0-score_1)])
        
        self.indices_diff_scores.sort(key= lambda x: x[1], reverse=True)
        self.indices_diff_scores = self.indices_diff_scores[:150000]
        self.grouped_datasets = []
        intervals = np.linspace(0, len(self.indices_diff_scores), groups+1).astype(np.int32)
        self.indices_diff_scores = np.array(self.indices_diff_scores)
        self.datasets_sizes = []
        for right_limit in intervals[1:]:
            # print(self.indices_diff_scores[:right_limit, 0])
            self.grouped_datasets.append(self.used_dataset.select(self.indices_diff_scores[:right_limit, 0]))
            self.datasets_sizes.append(self.grouped_datasets[-1].num_rows)
        self.current_group_idx = 0
        self.current_group = self.grouped_datasets[self.current_group_idx]
        # self.current_group.save_to_disk("/home/fl488644/Curriculum-DPO/datasets/pickapic_150k")
        self.split = split
        self.transform=None
        prompt_ds = load_dataset(data_dir,cache_dir=None, split='test', keep_in_memory=False)
        self.prompts = [caption for caption in prompt_ds["caption"]]
        self.usable_prompts = self.prompts

    def update_cl(self):
        self.current_group_idx +=1
        self.current_group = self.grouped_datasets[min(self.current_group_idx, len(self.grouped_datasets)-1)]

    def __len__(self):
        return self.datasets_sizes[min(self.current_group_idx, len(self.grouped_datasets)-1)]
    
    def __getitem__(self, idx):
        ds = self.current_group[idx]
        image_0_uid = ds["image_0_uid"]
        image_1_uid = ds["image_1_uid"]
        label_0 = ds['label_0']
        label_1 = ds['label_1']
        image_0 = Image.open(io.BytesIO(ds['jpg_0']))
        image_1 = Image.open(io.BytesIO(ds['jpg_1']))
        if label_0>label_1:
            image_w = image_0
            image_l = image_1
        else:
            image_w = image_1
            image_l = image_0
        prompt = ds["caption"]
        if self.transform is not None:
            return self.transform(image_w, image_l, prompt)
        return image_w, image_l, prompt
       
    def set_transform(self, transform):
        self.transform = transform        
def get_dataset(args, logger):
    if args.dataset=='pickapic':
        return PickaPic(data_dir = args.data_dir, score_dir=args.score_dir, split=args.subset, groups=args.no_chunks, reward=args.reward_fn)
    else:
        return PromptsDataset(logger, data_dir = args.data_dir, score_dir=args.score_dir, split=None, groups=args.no_chunks, reward=args.reward_fn)

if __name__ == "__main__":
    # dataset = PromptsDataset(data_dir="datasets/drawbench/test/sd", score_dir="scores/human",
    #              groups=1, split=None, reward='human_score')
    # image_1, image_2, prompt = dataset[10]
    # image_1.save("sample_1.png")
    # image_2.save("sample_2.png")
    # print(prompt)
    dataset = PickaPic(data_dir='/home/fl488644/datasets/pickapic', score_dir="/home/fl488644/Curriculum-DPO/scores/pick_a_pic_scores_human",
                 split="train", groups = 1, reward="human_score")
    image_1, image_2, prompt = dataset[10]
    image_1.save("sample_1.png")
    image_2.save("sample_2.png")
    print(prompt)
