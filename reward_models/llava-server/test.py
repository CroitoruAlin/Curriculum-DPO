import requests
from PIL import Image
import io
import pickle
import glob
import tqdm
from concurrent.futures import ThreadPoolExecutor

BATCH_SIZE = 18

# paths = glob.glob(f"target_paired_images_new_toykitchen7/*.jpg") * 100
paths = ["cat.png"] * 2
answers = [["a cat plays with a feather while sitting on a couch"], ["a small dog bargs to a person"]]
queries= [["what happens in the image?"], ["what happens in the image?"]]
def f(_):
    for i in tqdm.tqdm(range(0, len(paths), BATCH_SIZE)):
        batch_paths = paths[i : i + BATCH_SIZE]
        print(batch_paths)
        jpeg_data = []
        # queries = []
        for path in batch_paths:
            image = Image.open(path)

            # Compress the images using JPEG
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80)
            jpeg_data.append(buffer.getvalue())
            # print(jpeg_data)

            # queries.append(
            #     ["what happens in the image?"]
            # )
            # texts.append(["What item is lying on the table?"])
        print(queries)
        data = {"images": jpeg_data, "queries": queries, "answers": answers}
        data_bytes = pickle.dumps(data)

        # Send the JPEG data in an HTTP POST request to the server
        url = "http://127.0.0.1:8085"
        response = requests.post(url, data=data_bytes)

        # Print the response from the server
        response_data = pickle.loads(response.content)

        for output, score in zip(response_data["outputs"], response_data["recall"]):
            print(output)
            print(score)
            print("--")

with ThreadPoolExecutor(max_workers=8) as executor:
    for _ in executor.map(f, range(1)):
        pass