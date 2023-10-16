"""Text-to-image diffusion with GLIDE."""

from pathlib import Path
from PIL import Image
import torch
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
from tqdm import tqdm
from src import hierarchy
from src import prompt


class GlideGenerator:
    """A class for generating and saving diffusion images."""

    def __init__(self, images_per_synset: int, batch_size: int,
                 dataset_hierarchy: hierarchy.Hierarchy, prompt_name: str,
                 image_size: int, model_steps: str, up_steps: str,
                 guidance_scale: float, upsample_temp: float,
                 gpu_id: int, image_id_offset: int,
                 use_float16: bool):

        if not use_float16:
            print("Warning, this was not tested yet.")
        
        # Create base model.
        options = model_and_diffusion_defaults()
        options['use_fp16'] = use_float16
        options['timestep_respacing'] = model_steps
        self.options = options

        device = f"cuda:{gpu_id}"
        self.device = device
        self.image_id_offset = image_id_offset

        self.model, self.diffusion = create_model_and_diffusion(**options)
        self.model.eval()
        if use_float16:
            self.model.convert_to_fp16()
        self.model.to(device)
        self.model.load_state_dict(load_checkpoint('base', device))

        # Create upsampler model.
        options_up = model_and_diffusion_defaults_upsampler()
        options_up['use_fp16'] = use_float16
        options_up['timestep_respacing'] = up_steps
        self.options_up = options_up

        self.model_up, self.diffusion_up = create_model_and_diffusion(**options_up)
        self.model_up.eval()
        if use_float16:
            self.model_up.convert_to_fp16()
        self.model_up.to(device)
        self.model_up.load_state_dict(load_checkpoint('upsample', device))

        if images_per_synset % batch_size != 0:
            raise ValueError("The number of images per synset "
                             "should be divisible by batch size.")

        self.images_per_synset = images_per_synset
        self.batch_size = batch_size
        self.dataset_hierarchy = dataset_hierarchy
        self.prompt_fn = getattr(prompt, prompt_name)

        self.image_size = image_size
        self.guidance_scale = guidance_scale
        self.upsample_temp = upsample_temp

    def _tensor_to_pil_list(self, samples_tensor):
        samples = []

        for sample in samples_tensor:
            scaled = ((sample + 1) * 127.5).round().clamp(0,255).to(torch.uint8).cpu()
            reshaped = scaled.permute(1, 2, 0)
            samples.append(Image.fromarray(reshaped.numpy()))

        return samples

    def generate_prompt_images(self, prompt):
        """
        Generate images for the given prompt.
        """
        # Create the text tokens to feed to the model.
        tokens = self.model.tokenizer.encode(prompt)
        tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
            tokens, self.options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = self.batch_size * 2
        uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask(
            [], self.options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=torch.tensor(
                [tokens] * self.batch_size + [uncond_tokens] * self.batch_size,
                device=self.device
            ),
            mask=torch.tensor(
                [mask] * self.batch_size + [uncond_mask] * self.batch_size,
                dtype=torch.bool,
                device=self.device,
            ),
        )

        # Create a classifier-free guidance sampling function
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        # Sample from the base model.
        self.model.del_cache()
        samples = self.diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, self.options["image_size"], self.options["image_size"]),
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:self.batch_size]
        self.model.del_cache()

        tokens = self.model_up.tokenizer.encode(prompt)
        tokens, mask = self.model_up.tokenizer.padded_tokens_and_mask(
            tokens, self.options_up['text_ctx']
        )

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples+1)*127.5).round()/127.5 - 1,

            # Text tokens
            tokens=torch.tensor(
                [tokens] * self.batch_size, device=self.device
            ),
            mask=torch.tensor(
                [mask] * self.batch_size,
                dtype=torch.bool,
                device=self.device,
            ),
        )

        # Sample from the base model.
        self.model_up.del_cache()
        up_shape = (self.batch_size, 3, self.options_up["image_size"], self.options_up["image_size"])
        up_samples = self.diffusion_up.ddim_sample_loop(
            self.model_up,
            up_shape,
            noise=torch.randn(up_shape, device=self.device) * self.upsample_temp,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:self.batch_size]
        self.model_up.del_cache()
        
        return self._tensor_to_pil_list(up_samples)

    def generate_synset_images(self, images_path: str,
                               synset: hierarchy.Synset):
        """
        Generates images of a given synset and saves them to `images_path`.
        """

        synset_path = Path(images_path) / synset.name()
        if synset_path.is_dir() and (synset_path / "31.png").is_file():
            print(f"Skipping {synset.name()}, directory already exists and has images.")
            return

        synset_path.mkdir(parents=True)

        for batch_id in range(self.images_per_synset // self.batch_size):
            synset_prompt = self.prompt_fn(synset)

            images = self.generate_prompt_images(synset_prompt)
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
        for synset in tqdm(self.dataset_hierarchy.get_all_synsets(remove_leaves)):
            self.generate_synset_images(images_path, synset)

    def generate_subtree(self, images_path: str,
                         parent_synset: hierarchy.Synset, remove_leaves: bool):
        """
        Generates images of subtree synsets and saves them to `images_path`.
        """
        for synset in self.dataset_hierarchy.get_synset_subtree(parent_synset,
                                                                remove_leaves):
            self.generate_synset_images(images_path, synset)

