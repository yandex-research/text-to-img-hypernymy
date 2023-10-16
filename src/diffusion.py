"""Text-to-image diffusion."""

from pathlib import Path
import diffusers
import torch
from src import hierarchy
from src import prompt


def dummy_checker_fn(images, **kwargs):
    return images, False


class DiffusionGenerator:
    """A class for generating and saving diffusion images."""

    def __init__(self, model_name: str, images_per_synset: int, batch_size: int,
                 dataset_hierarchy: hierarchy.Hierarchy, prompt_name: str,
                 image_size: int, num_inference_steps: int,
                 guidance_scale: float, eta: float,
                 gpu_id: int, image_id_offset: int,
                 use_float16: bool):

        self.diffusion = diffusers.DiffusionPipeline.from_pretrained(
            model_name, torch_dtype=(torch.float16 if use_float16 else None))
        self.diffusion.safety_checker = dummy_checker_fn

        # Hard-coded for unCLIP.
        self.is_unclip = (model_name == "kakaobrain/karlo-v1-alpha")

        self.diffusion = self.diffusion.to(f"cuda:{gpu_id}")
        self.image_id_offset = image_id_offset

        if images_per_synset % batch_size != 0:
            raise ValueError("The number of images per synset "
                             "should be divisible by batch size.")

        self.images_per_synset = images_per_synset
        self.batch_size = batch_size
        self.dataset_hierarchy = dataset_hierarchy
        self.prompt_fn = getattr(prompt, prompt_name)

        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.eta = eta

    def generate_synset_images(self, images_path: str,
                               synset: hierarchy.Synset):
        """
        Generates images of a given synset and saves them to `images_path`.
        """

        synset_path = Path(images_path) / synset.name()
        synset_path.mkdir(parents=True)

        for batch_id in range(self.images_per_synset // self.batch_size):
            synset_prompt = self.prompt_fn(synset)
            prompts_batch = [synset_prompt] * self.batch_size

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
                absolute_image_id = batch_id * self.batch_size + image_id + self.image_id_offset
                image_path = synset_path / f"{absolute_image_id}.png"
                image.save(image_path.resolve())

    def generate_images(self, images_path: str, remove_leaves: bool):
        """
        Generates images of all the synsets and saves them to `images_path`.
        """
        for synset in self.dataset_hierarchy.get_all_synsets(remove_leaves):
            self.generate_synset_images(images_path, synset)

    def generate_subtree(self, images_path: str,
                         parent_synset: hierarchy.Synset, remove_leaves: bool):
        """
        Generates images of subtree synsets and saves them to `images_path`.
        """
        for synset in self.dataset_hierarchy.get_synset_subtree(parent_synset,
                                                                remove_leaves):
            self.generate_synset_images(images_path, synset)
