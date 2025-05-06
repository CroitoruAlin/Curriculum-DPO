# Curriculum-DPO
Curriculum Direct Preference Optimization for Diffusion and Consistency Models


<a href='https://croitorualin.github.io/cl-dpo/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/pdf/2405.13637'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

## Environment
```{bash}
conda create -n dpo python=3.9
conda activate dpo
pip install -r requirements.txt
```

## Inference

### Weights

```{bash}
git lfs install
git clone https://huggingface.co/acroitoru/curriculum-dpo-loras
```
### Run inference on a single example
We have different scripts for SD (``inference_sd.py``) and LCM (``inference_lcm.py``). For example, you can run inference using a LCM ckpt as follows:
```{bash}
python inference_lcm.py --lora_ckpt curriculum-dpo-loras/<choose the ckpt you want> --prompt "A red dog"
```
### Run inference on a dataset
We have separate scripts for generating an entire dataset under the ``data_generators`` folder. In addition to the lora ckpt, we recommend setting: the source of the prompts through the parameter ``--dataset``, the batch_size ``--sample_batch_size`` and the reward function ``--reward_fn``. Please note that the latter does not have a functional role, it is only used to create the folder structure where the generated images are saved. Example:
```{bash}
python data_generators/generate_data_lcm.py --lora_ckpt curriculum-dpo-loras/lcm_animals_text.safetensors  --dataset animals --reward_fn llavabertscore --sample_batch_size 20
```
### Run reward models over a dataset
The scripts for running the reward models are in the ``reward_models`` folder. Each script receives as parameter the path to the dataset that needs evaluation.
```{bash}
python reward_models/compute_human_score.py --dataset_path <dataset_path>
```
#### LLaVA server
The text alignement evaluation is more complex to run because it needs a llava server.
We followed this github repo https://github.com/kvablack/LLaVA-server to setup this server.
First, create a new conda environment:
```{bash}
conda create -n llava-server python=3.8
```
Then change the current directory to ``reward_models/llava-server`` and run:
```{bash}
pip install -r requirements.txt
```
Finally, run:
```{bash}
gunicorn "app:create_app()" &
```
Now you can run ``compute_text_align_score.py`` as previously described.
!!! Note that ``gunicorn "app:create_app()" &`` requires the llava ckpt to be stored under ``llava-v1.5-13b`` in pwd.

## Train

### Preliminaries
1. Before training, you need to run the image generation if the data set contains only prompts (as is the case for animals and drawbench). You can do this by following the instructions from the inference section, but please omit the parameter ``--lora_ckpt`` and set ``--subset`` to ``train`` because it changes the number of generated samples from 10 to 500.

You can skip this step, if you are using a dataset with images, like a Pick-a-pic.

2. Run the reward model to be able to decide the preferences and the curriculum. Follow the instructions from the inference section.
### Train
LCM:
```{bash}
accelerate launch train_lcm.py --data_dir <dataset_path> --score_dir <path_to_the_folder_where_the_scores_are_stored> --dataset <type_of_dataset> # type_of_dataset can be drawbench/animals/pickapic
```

```{bash}
accelerate launch train_stablediffusion.py --data_dir <dataset_path> --score_dir <path_to_the_folder_where_the_scores_are_stored> --dataset <type_of_dataset> # type_of_dataset can be drawbench/animals/pickapic
```

## Citation
```
@inproceedings{Croitoru-CVPR-2025,
      title={Curriculum Direct Preference Optimization for Diffusion and Consistency Models}, 
      author={Florinel Alin Croitoru and Vlad Hondru and Radu Tudor Ionescu and Nicu Sebe and Mubarak Shah},
      year={2025},
      booktitle={Proceedings of CVPR}
}
```