
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from PIL import Image
import io
import joblib
from tqdm import tqdm


class PickaPic(Dataset):

    def __init__(self, data_dir='/mnt/home/fmi2/alin/datasets/pickapic_v1', score_dir="/home/fl488644/Curriculum-DPO/scores/pick_a_pic_scores_human",
                  return_uids=False, split="train", preprocess_images = None, preprocess_text = None, groups = 5):
        self.dataset = load_dataset("yuvalkirstain/pickapic_v1", cache_dir=data_dir, keep_in_memory=False)
        self.scores = load_from_disk(score_dir, keep_in_memory=True)
        scores_images_0 = np.array([score for score in self.scores['image_0_score']])
        scores_images_1 = np.array([score for score in self.scores['image_1_score']])
        orig_len = self.dataset[split].num_rows
        self.original_dataset = self.dataset[split]
        self.original_dataset = self.original_dataset.add_column(name="image_0_score", column = [score for score in self.scores['image_0_score']])
        self.original_dataset = self.original_dataset.add_column(name="image_1_score", column = [score for score in self.scores['image_1_score']])
        not_split_idx = []
        for i, label_0 in tqdm(enumerate(self.original_dataset['label_0'])):
            if label_0 in (0,1) and abs(scores_images_0[i] - scores_images_1[i])>0.01 and(scores_images_0[i]>0.1 and scores_images_1[i]>0.1):
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
            self.grouped_datasets.append(self.used_dataset.select(self.indices_diff_scores[:right_limit, 0]))
            self.datasets_sizes.append(self.grouped_datasets[-1].num_rows)
        self.current_group_idx = 0
        self.current_group = self.grouped_datasets[self.current_group_idx]
        self.return_uids = return_uids
        self.split = split
        self.preprocess_images = preprocess_images
        self.preprocess_text = preprocess_text
        self.transform=None
        self.prompts = [caption for caption in self.dataset['test']["caption"]]

    def update_cl(self):
        self.current_group_idx +=1
        self.current_group = self.grouped_datasets[min(self.current_group_idx, len(self.grouped_datasets)-1)]

    def __len__(self):
        return self.datasets_sizes[min(self.current_group_idx, len(self.grouped_datasets)-1)]
    
    def __getitem__(self, idx):
        ds = self.current_group[idx]
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
    