from PIL import Image
import io
import numpy as np
import torch
import time
from PIL import Image
import io
import numpy as np
import torch
import time

def aesthetic_score():
    from utils.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 10
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            if images.shape[1] ==3 or images.shape[1]==3:
                images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]] * len(image_batch),
                "answers": [[f"The image contains {prompt}"] for prompt in prompt_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            not_respone =True
            while not_respone:
                try:
                    response = sess.post(url, data=data_bytes, timeout=120)
                    not_respone=False
                except:
                    print("Failed to reach LLAVA. Sleeping 1 minute")
                    time.sleep(60)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            try:
                all_scores += scores.tolist()
                # save the precision and f1 scores for analysis
                all_info["precision"] += np.array(response_data["precision"]).squeeze().tolist()
                all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
                all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()
            except:
                all_scores += [scores]
                all_info["precision"] += [np.array(response_data["precision"]).squeeze()]
                all_info["f1"] += [np.array(response_data["f1"]).squeeze()]
                all_info["outputs"] += [np.array(response_data["outputs"]).squeeze()]
            # print(response_data)
            # all_scores += scores

            
        # print(all_scores)
        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn
