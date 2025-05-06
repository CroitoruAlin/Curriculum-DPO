from torchvision import transforms
import torch
class Preprocess:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.train_transforms = transforms.Compose(
        [
            transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.resolution) if config.center_crop else transforms.RandomCrop(config.resolution),
            transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def tokenize_captions(self,examples, is_train=True):
        captions = []
        for caption in examples:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    # Preprocessing the datasets.
    
    def preprocess_train(self, images_w, images_l, captions):
        examples = {}
        images_w = [image.convert("RGB") for image in images_w]
        examples["pixel_values_w"] = [self.train_transforms(image) for image in images_w]
        images_l = [image.convert("RGB") for image in images_l]
        examples["pixel_values_l"] = [self.train_transforms(image) for image in images_l]
        examples["input_ids"] = self.tokenize_captions(captions)
        # print(captions)
        return examples
    def collate_fn(self, examples):
        images_w = [example[0] for example in examples]
        images_l = [example[1] for example in examples]
        captions = [example[2] for example in examples]
        examples = self.preprocess_train(images_w, images_l, captions)
        pixel_values_w = torch.stack(examples['pixel_values_w'])
        pixel_values_w = pixel_values_w.to(memory_format=torch.contiguous_format).float()
        pixel_values_l = torch.stack(examples['pixel_values_l'])
        pixel_values_l = pixel_values_l.to(memory_format=torch.contiguous_format).float()
        
        input_ids = examples["input_ids"]
        return {"pixel_values_w": pixel_values_w, "pixel_values_l": pixel_values_l, "input_ids": captions}
    def preprocess_lcm_train(self, images, captions):
        examples = {}
        images = [image.convert("RGB") for image in images]
        examples["pixel_values"] = [self.train_transforms(image) for image in images]
        examples["input_ids"] = self.tokenize_captions(captions)
        return examples
    def collate_lcm_fn(self, examples):
        images = [example[0] for example in examples]
        captions = [example[1] for example in examples]
        examples = self.preprocess_lcm_train(images, captions)
        pixel_values = torch.stack(examples['pixel_values'])
        
        return {"pixel_values": pixel_values, "input_ids": captions}