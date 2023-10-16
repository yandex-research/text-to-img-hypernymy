"""Image generation script."""

from pathlib import Path
from absl import app, flags
from src import coco_glide
from src import hierarchy


FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", None,
                     help="Batch size for generation.", required=True, lower_bound=1)
flags.DEFINE_integer("image_size", None,
                     help="Size (height, width) of generated images.", required=True)

flags.DEFINE_integer("num_images", None,
                     help="Number of images to generate.", required=True, lower_bound=1)

# Diffusion sampler parameters.
flags.DEFINE_string("model_steps", "50", help="Number of model steps.")
flags.DEFINE_string("up_steps", "fast27", help="Number of model steps.")

flags.DEFINE_float("guidance_scale", 3.0, help="Diffusion guidance scale.")
flags.DEFINE_float("upsample_temp", 0.997, help="Upsample temperature.")

flags.DEFINE_string("images_path", None,
                    help="Path for saving the images.", required=True)

flags.DEFINE_integer("gpu_id", None, "Gpu id.", required=True)
flags.DEFINE_integer("image_id_offset", 0,
                     "Image save path offset for multi-gpu generation.")

flags.DEFINE_bool("use_float16", None, "If true uses float16.", required=True)


def main(_):
    """Script body."""
    diffusion_coco_generator = coco_glide.GlideCocoGenerator(
        FLAGS.num_images, FLAGS.batch_size,
        FLAGS.image_size,
        FLAGS.model_steps, FLAGS.up_steps, FLAGS.guidance_scale,
        FLAGS.upsample_temp,
        FLAGS.gpu_id, FLAGS.image_id_offset, FLAGS.use_float16)

    diffusion_coco_generator.generate_images(
        images_path=FLAGS.images_path)


if __name__ == "__main__":
    app.run(main)
