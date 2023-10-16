"""Image generation script."""

from pathlib import Path
from absl import app, flags
from src import coco
from src import hierarchy


FLAGS = flags.FLAGS
flags.DEFINE_string("model_name", None,
                    help="Diffusers pre-trained model name.", required=True)
flags.DEFINE_integer("batch_size", None,
                     help="Batch size for generation.", required=True, lower_bound=1)
flags.DEFINE_integer("image_size", None,
                     help="Size (height, width) of generated images.", required=True)

flags.DEFINE_integer("num_images", None,
                     help="Number of images to generate.", required=True, lower_bound=1)

# Diffusion sampler parameters.
flags.DEFINE_integer("num_inference_steps", 10,
                     help="Number of diffusion steps.", lower_bound=1)
flags.DEFINE_float("guidance_scale", 7.5, help="Diffusion guidance scale.")
flags.DEFINE_float("eta", 0,
                   help="Eta, DDIM if 0, DDPM if 1.", lower_bound=0, upper_bound=1)

flags.DEFINE_string("images_path", None,
                    help="Path for saving the images.", required=True)

flags.DEFINE_integer("gpu_id", None, "Gpu id.", required=True)
flags.DEFINE_integer("image_id_offset", 0,
                     "Image save path offset for multi-gpu generation.")

flags.DEFINE_bool("use_float16", None, "If true uses float16.", required=True)


def main(_):
    """Script body."""
    diffusion_coco_generator = coco.DiffusionCocoGenerator(
        FLAGS.model_name, FLAGS.num_images, FLAGS.batch_size,
        FLAGS.image_size,
        FLAGS.num_inference_steps, FLAGS.guidance_scale, FLAGS.eta,
        FLAGS.gpu_id, FLAGS.image_id_offset, FLAGS.use_float16)

    diffusion_coco_generator.generate_images(
        images_path=FLAGS.images_path)


if __name__ == "__main__":
    app.run(main)
