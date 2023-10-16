"""Image generation script."""

from pathlib import Path
from absl import app, flags
from src import diffusion
from src import hierarchy


FLAGS = flags.FLAGS
flags.DEFINE_string("model_name", None,
                    help="Diffusers pre-trained model name.", required=True)
flags.DEFINE_string("prompt_name", "base_prompt",
                    help="Prompting function name (see src/prompt.py).")
flags.DEFINE_list("synsets", [],
                  help="Synset WNIDs for subtree generation.")
flags.DEFINE_integer("images_per_synset", None,
                     help="Number of generated images per synset.", required=True, lower_bound=1)
flags.DEFINE_bool("remove_leaves", None,
                  "Whether to remove leaf synsets from generation.", required=True)
flags.DEFINE_integer("batch_size", None,
                     help="Batch size for generation.", required=True, lower_bound=1)
flags.DEFINE_integer("image_size", None,
                     help="Size (height, width) of generated images.", required=True)

# Dataset.
flags.DEFINE_enum("dataset_name", "imagenet", ["imagenet"],
                  "Dataset for the hierarchy.")

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
    dataset_classes_path = (Path(__file__).parents[1] /
                            "wordnet_classes" /
                            FLAGS.dataset_name).with_suffix(".txt")
    dataset_hierarchy = hierarchy.Hierarchy(dataset_classes_path.resolve())
    diffusion_generator = diffusion.DiffusionGenerator(
        FLAGS.model_name, FLAGS.images_per_synset, FLAGS.batch_size,
        dataset_hierarchy, FLAGS.prompt_name, FLAGS.image_size,
        FLAGS.num_inference_steps, FLAGS.guidance_scale, FLAGS.eta,
        FLAGS.gpu_id, FLAGS.image_id_offset, FLAGS.use_float16)

    if len(FLAGS.synsets) != 0:
        for synset in FLAGS.synsets:
            diffusion_generator.generate_synset_images(
                images_path=FLAGS.images_path,
                synset=dataset_hierarchy.get_synset_from_name(synset))
    else:
        diffusion_generator.generate_images(
            images_path=FLAGS.images_path,
            remove_leaves=FLAGS.remove_leaves)


if __name__ == "__main__":
    app.run(main)
