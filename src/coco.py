"""
Text-to-image diffusion on COCO.

Metrics on COCO were computed using pytorch-fid [1] and torchmetrics [2]
for FID and CLIPScore, respectively.

[1] - https://github.com/mseitzer/pytorch-fid
[2] - https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_score.html
"""

import random
import json
from pathlib import Path
import tqdm
import diffusers
import torch


def dummy_checker_fn(images, **kwargs):
    return images, False


class DiffusionCocoGenerator:
    """A class for generating and saving diffusion images."""

    def __init__(self, model_name: str, num_images: int, batch_size: int,
                 image_size: int, num_inference_steps: int,
                 guidance_scale: float, eta: float,
                 gpu_id: int, image_id_offset: int,
                 use_float16: bool,
                 # Hard coded path for our experiments.
                 captions_path: str = "YOUR_PATH/captions_val2014.json"):
        self.diffusion = diffusers.DiffusionPipeline.from_pretrained(
            model_name, torch_dtype=(torch.float16 if use_float16 else None))
        self.diffusion.safety_checker = dummy_checker_fn

        # Hard-coded for unCLIP.
        self.is_unclip = (model_name == "kakaobrain/karlo-v1-alpha")

        self.diffusion = self.diffusion.to(f"cuda:{gpu_id}")
        self.image_id_offset = image_id_offset

        if self.image_id_offset != 0:
            raise NotImplementedError("Multi-gpu generation is not implemented yet, "
                                      "image_id_offset should be set to 0.")

        if num_images % batch_size != 0:
            raise ValueError("The number of images "
                             "should be divisible by batch size.")

        self.captions_path = captions_path
        self.num_images = num_images
        self.batch_size = batch_size

        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.eta = eta

    def generate_images(self, images_path: str, seed: int = 0):
        """
        Generates images of all the synsets and saves them to `images_path`.
        """
        # Ensures that all captions are the same across models.
        random.seed(seed)

        save_path = Path(images_path)
        save_path.mkdir()

        with open(self.captions_path) as file:
            annotations = json.load(file)["annotations"]
        captions = [x["caption"] for x in annotations]
        captions_subset = random.sample(captions, self.num_images)

        for batch_id in tqdm.tqdm(range(self.num_images // self.batch_size)):
            start_id = batch_id * self.batch_size
            prompts_batch = captions_subset[start_id:start_id + self.batch_size]

            default_kwargs = {
                "guidance_scale": self.guidance_scale,
                "eta": self.eta,
                "num_inference_steps": self.num_inference_steps,
            }

            unclip_kwargs = {
                "decoder_guidance_scale": self.guidance_scale,
                "prior_guidance_scale": self.guidance_scale,
            }

            diffusion_kwargs = default_kwargs if not self.is_unclip else unclip_kwargs

            if batch_id == 0 and self.is_unclip:
                print("WARNING: eta and num_inference_steps are ignored for unCLIP.")

            images = self.diffusion(
                prompts_batch, **diffusion_kwargs).images
            
            images = [image.resize((self.image_size, self.image_size))
                      for image in images]

            for image_id, image in enumerate(images):
                absolute_image_id = start_id + image_id
                image_path = save_path / f"{absolute_image_id}.png"
                image.save(image_path.resolve())
