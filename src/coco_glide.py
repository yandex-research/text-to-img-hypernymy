"""Text-to-image diffusion on COCO with GLIDE."""

import random
import json
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


class GlideCocoGenerator:
    """A class for generating and saving diffusion images."""

    def __init__(self, num_images: int, batch_size: int,
                 image_size: int, model_steps: str, up_steps: str,
                 guidance_scale: float, upsample_temp: float,
                 gpu_id: int, image_id_offset: int,
                 use_float16: bool,
                 captions_path: str = "YOUR_PATH/captions_val2014.json"):

        if not use_float16:
            print("Warning, this was not tested yet.")

        if num_images % batch_size != 0:
            raise ValueError("The number of images "
                             "should be divisible by batch size.")

        self.captions_path = captions_path
        self.num_images = num_images
        self.batch_size = batch_size
        
        # Create base model.
        options = model_and_diffusion_defaults()
        options['use_fp16'] = use_float16
        options['timestep_respacing'] = model_steps
        self.options = options

        device = f"cuda:{gpu_id}"
        self.device = device
        self.image_id_offset = image_id_offset

        if self.image_id_offset != 0:
            raise NotImplementedError("Multi-gpu generation is not implemented yet, "
                                      "image_id_offset should be set to 0.")

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

    def generate_prompt_images(self, prompts_batch):
        """
        Generate images for the given prompt.
        """
        glide = self.model

        tokens = [glide.tokenizer.encode(p) for p in prompts_batch]
        tokens_and_masks = [glide.tokenizer.padded_tokens_and_mask(t, self.options['text_ctx']) for t in tokens]
        tokens = [t for t,_ in tokens_and_masks]
        masks = [m for _,m in tokens_and_masks]

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = self.batch_size * 2
        uncond_tokens, uncond_mask = glide.tokenizer.padded_tokens_and_mask([], self.options['text_ctx'])


        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=torch.tensor(
                tokens + [uncond_tokens] * self.batch_size,
                device=self.device
            ),
            mask=torch.tensor(
                masks + [uncond_mask] * self.batch_size,
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

        glide = self.model_up
        tokens = [glide.tokenizer.encode(p) for p in prompts_batch]
        tokens_and_masks = [glide.tokenizer.padded_tokens_and_mask(t, self.options_up['text_ctx']) for t in tokens]
        tokens = [t for t,_ in tokens_and_masks]
        masks = [m for _,m in tokens_and_masks]

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples+1)*127.5).round()/127.5 - 1,

            # Text tokens
            tokens=torch.tensor(
                tokens, device=self.device
            ),
            mask=torch.tensor(
                masks,
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

        for batch_id in tqdm(range(self.num_images // self.batch_size)):
            start_id = batch_id * self.batch_size
            prompts_batch = captions_subset[start_id:start_id + self.batch_size]

            images = self.generate_prompt_images(prompts_batch)
            
            images = [image.resize((self.image_size, self.image_size))
                      for image in images]

            for image_id, image in enumerate(images):
                absolute_image_id = start_id + image_id
                image_path = save_path / f"{absolute_image_id}.png"
                image.save(image_path.resolve())
