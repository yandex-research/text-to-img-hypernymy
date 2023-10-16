"""Metric computation script."""

from pathlib import Path
from absl import app, flags
from src import metrics
from src import hierarchy


FLAGS = flags.FLAGS
flags.DEFINE_string("logits_path", None,
                    help="Path to the calculated metrics.", required=True)
flags.DEFINE_list("metric_names", None,
                  "List of metric class names (e.g. SubtreeEntropy) to calculate.", required=True)
flags.DEFINE_string("metrics_path", None,
                    help="Path to the calculated metrics.", required=True)
flags.DEFINE_integer("num_samples", None,
                     help="Number of samples for metrics computation.", required=False)

# Dataset.
flags.DEFINE_enum("dataset_name", "imagenet", ["imagenet"],
                  "Dataset for the hierarchy.")


def main(_):
    """Script body."""
    dataset_classes_path = (Path(__file__).parents[1] /
                            "wordnet_classes" /
                            FLAGS.dataset_name).with_suffix(".txt")
    dataset_hierarchy = hierarchy.Hierarchy(dataset_classes_path.resolve())

    for metric_name in FLAGS.metric_names:
        if not hasattr(metrics, metric_name):
            raise ValueError(f"Metric class {metric_name} doesn't exist.")

    metric_classes = [
        getattr(metrics, metric_name) for metric_name in FLAGS.metric_names]
    metric_list = [
        metric_clas(dataset_hierarchy, num_samples=FLAGS.num_samples)
        for metric_clas in metric_classes]

    metric_set = metrics.MetricSet(*metric_list)
    metric_set.compute_and_save_metrics(FLAGS.logits_path, FLAGS.metrics_path)

if __name__ == "__main__":
    app.run(main)
