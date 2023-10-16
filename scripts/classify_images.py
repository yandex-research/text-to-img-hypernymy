"""Image classification script."""

from absl import app, flags
from src import classification


FLAGS = flags.FLAGS
flags.DEFINE_list("model_name", None,
                    help="Classifier torchvision model names.", required=True)
flags.DEFINE_string("images_path", None,
                    help="Path to the generated images.", required=True)
flags.DEFINE_string("logits_path", None,
                    help="Path for saving the logits.", required=True)
flags.DEFINE_integer("batch_size", None,
                     help="Batch size.", required=True)

flags.DEFINE_integer("gpu_id", None, "Gpu id.", required=True)


def main(_):
    """Script body."""
    for model_name in FLAGS.model_name:
        classifier = classification.Classifier(
            model_name, FLAGS.batch_size, FLAGS.gpu_id)
        classifier.compute_logits(FLAGS.images_path, FLAGS.logits_path)


if __name__ == "__main__":
    app.run(main)
