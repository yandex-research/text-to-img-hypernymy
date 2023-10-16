"""COCO CLIP score."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchmetrics.multimodal import CLIPScore
from tqdm import tqdm


@torch.no_grad()
def get_clip_score(images_path, prompts, clip_score, num_images, gpu_id, batch_size=32):
    clip_scores = []

    all_images = list(images_path.glob("*"))
    all_images = sorted(all_images, key=lambda x: int(str(x.stem)))

    total_prompts = 0

    for start in tqdm(range(0, num_images, batch_size)):
        img = all_images[start:min(start + batch_size, num_images)]
        img = [TF.to_tensor(Image.open(x)).to(f"cuda:{gpu_id}") for x in img]

        prompt = prompts[start:min(start + batch_size, num_images)]

        clip_scores.append(clip_score(img, prompt).item() / 100 * len(prompt))
        total_prompts += len(prompt)

        if start % 10 == 0:
            print(start, "CLIP score", sum(clip_scores) / total_prompts)
    
    assert total_prompts == num_images
    
    print("CLIP score", sum(clip_scores) / num_images)
    return sum(clip_scores) / num_images, clip_scores


def main():
    _, images_path, gpu_id_str = sys.argv
    images_path = Path(images_path)
    gpu_id = int(gpu_id_str)

    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(f"cuda:{gpu_id}")

    # Numpy array of selected prompts, hard-coded for our experiments.
    prompts = list(np.load("coco_prompts.npy"))
    print(prompts[:10])

    average_score, scores = get_clip_score(
        images_path, prompts, clip_score, 10000, gpu_id,
    )

    print(average_score)
    np.save(f"scores_{images_path.stem}.npy", np.array(scores))


if __name__ == "__main__":
    main()
