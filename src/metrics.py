"""Computing hierarchy metrics."""

import abc
import typing
from pathlib import Path
import numpy as np
import scipy
from src import hierarchy


MetricValues = typing.Dict[str, typing.Any]
FilterFn = typing.Callable[[str], bool]


class LogitsSlice:
    """Used to load a fixed number of sample logits."""

    def __init__(self, logits_path: str,
                 num_samples: typing.Optional[int]):
        self.logits_path = logits_path
        self.loaded_logits = np.load(logits_path)
        self.num_samples = num_samples

    def __getitem__(self, key: str):
        logits = self.loaded_logits[key]

        num_samples = (logits.shape[0] if self.num_samples is None
                       else self.num_samples)

        if logits.shape[0] < num_samples:
            raise ValueError(
                f"Logits loaded from {self.logits_path} have"
                f"{logits.shape[0]} samples at {key} which is less"
                f"than {self.num_samples}")

        return logits[:num_samples]

    def items(self) -> typing.Iterator[typing.Tuple[str, np.ndarray]]:
        """Iterates over `(synset_name, logits)` pairs."""
        for key in self.loaded_logits:
            yield (key, self[key])


class HierarchyMetric(abc.ABC):
    """ABC for a hierarchy metric."""

    def __init__(self, dataset_hierarchy: hierarchy.Hierarchy,
                 num_samples: typing.Optional[int]) -> None:
        self.dataset_hierarchy = dataset_hierarchy
        self.num_samples = num_samples

    @property
    @abc.abstractmethod
    def base_name(self) -> str:
        """Base metric name (e.g. 'subtree_entropy')."""

    @property
    def name(self) -> str:
        """Metric name with the added number of samples if not None."""
        if self.num_samples is None:
            return self.base_name
        else:
            return f"{self.base_name}_{self.num_samples}"

    @abc.abstractmethod
    def compute_metric(self, logits_path: str) -> MetricValues:
        """Computes metric from saved logits at `logits_path`."""

    def save_metric(self, metric_values: MetricValues, metrics_path: str):
        """Saves metric values to `metrics_path` as a `.npz` file."""
        save_path = Path(metrics_path)
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f"{self.name}.npz"

        np.savez(save_path, **metric_values)

    def compute_and_save_metric(self, logits_path: str, metrics_path: str,
                                print_metric: bool = True):
        """
        Computes metric from saved logits at `logits_path` and saves values
        to `metrics_path` as a `.npz` file.
        """
        metric_values = self.compute_metric(logits_path)
        if print_metric:
            print(metric_values["average"])
        self.save_metric(metric_values, metrics_path)

    def load_logits(self, logits_path: str):
        """Loads logits from `logits_path`."""
        return LogitsSlice(logits_path, self.num_samples)


class MetricSet:
    """Computes a specified set of metrics."""

    def __init__(self, *metrics: HierarchyMetric):
        self.metrics = metrics

    def compute_metrics(self, logits_path):
        """Computes metrics from saved logits at `logits_path`."""
        metric_values_dict = {}

        for metric in self.metrics:
            metric_value = metric.compute_metric(logits_path)
            metric_values_dict[metric.name] = metric_value

        return metric_values_dict

    def save_metrics(self, metric_values_dict: typing.Dict[str, MetricValues],
                     metrics_path: str):
        """Saves metric values to `metrics_path` as separate `.npz` files"""
        save_path = Path(metrics_path)
        save_path.mkdir(parents=True, exist_ok=True)

        for metric_name, metric_values in metric_values_dict.items():
            metric_save_path = save_path / f"{metric_name}.npz"
            np.savez(metric_save_path, **metric_values)

    def compute_and_save_metrics(self, logits_path: str, metrics_path: str,
                                 print_metrics: bool = True):
        """
        Computes metrics from saved logits at `logits_path` and saves values
        to `metrics_path` as a `.npz` file.
        """
        metric_values_dict = self.compute_metrics(logits_path)
        if print_metrics:
            for metric_name, metric_value in metric_values_dict.items():
                print(metric_name, metric_value["average"])
        self.save_metrics(metric_values_dict, metrics_path)


class SubtreeMetric(HierarchyMetric):
    """ABC for computing independent subtree metrics."""

    def __init__(self, dataset_hierarchy: hierarchy.Hierarchy,
                 num_samples: typing.Optional[int],
                 filter_fn: typing.Optional[FilterFn] = None) -> None:
        super().__init__(dataset_hierarchy, num_samples)
        self.filter_fn = filter_fn

    @abc.abstractmethod
    def compute_subtree_metric(self, synset_name: str, logits: np.ndarray) -> typing.Any:
        """Computes metric independently for the given subtree."""

    def compute_metric(self, logits_path: str) -> typing.Any:
        logit_slices = self.load_logits(logits_path)
        metric_values = {}

        for synset_name, logits in logit_slices.items():
            if ((self.filter_fn is not None) and
                (not self.filter_fn(synset_name))):
                continue

            metric_values[synset_name] = self.compute_subtree_metric(synset_name, logits)

        subtree_metrics = list(metric_values.values())
        average_metric = sum(subtree_metrics) / len(subtree_metrics)
        metric_values["average"] = average_metric

        return metric_values


class SubtreeMoreThanOneChildMetric(SubtreeMetric):
    def __init__(self, dataset_hierarchy: hierarchy.Hierarchy,
        num_samples: typing.Optional[int],
        filter_fn: typing.Optional[FilterFn] = None) -> None:
        if filter_fn is not None:
            raise ValueError("SubtreeMoreThanOneChildMetric should have no filter_fn.")
        
        super().__init__(dataset_hierarchy, num_samples, filter_fn)

        self.synset_num_children = {}

        for synset in dataset_hierarchy.get_all_synsets(remove_leaves=True):
            n_ch = len(dataset_hierarchy.get_classifiable_subtree(synset))
            self.synset_num_children[synset.name()] = n_ch

        self.filter_fn = lambda synset_name: self.synset_num_children[synset_name] > 1


class SubtreeEntropy(SubtreeMoreThanOneChildMetric):
    """Computes subtree entropies."""

    base_name = "subtree_entropy"

    def compute_subtree_metric(self, synset_name: str, logits: np.ndarray) -> typing.Any:
        synset = self.dataset_hierarchy.get_synset_from_name(synset_name)
        classifiable_subtree = (self.dataset_hierarchy
                                .get_classifiable_subtree(synset))
        subtree_indices = [pair.class_id for pair in classifiable_subtree]
        subtree_logits = logits[:, subtree_indices]

        # [N x CLASS]
        subtree_probs = scipy.special.softmax(subtree_logits, axis=1)
        # [N x CLASS] -> [N]
        subtree_entropy = scipy.stats.entropy(subtree_probs, axis=1)
        # [N] -> []
        subtree_average_entropy = subtree_entropy.mean()
        return subtree_average_entropy


class SubtreeUniformEntropy(SubtreeEntropy):
    """Computes subtree entropies for uniform logits."""

    base_name = "subtree_uniform_entropy"

    def compute_subtree_metric(self, synset_name: str, logits: np.ndarray) -> typing.Any:
        uniform_logits = np.zeros_like(logits)
        return super().compute_subtree_metric(synset_name, uniform_logits)


class SubtreeMarginalEntropy(SubtreeMoreThanOneChildMetric):
    """Computes subtree entropies for the marginal distribution."""

    base_name = "subtree_marginal_entropy"

    def compute_subtree_metric(self, synset_name: str, logits: np.ndarray) -> typing.Any:
        synset = self.dataset_hierarchy.get_synset_from_name(synset_name)
        classifiable_subtree = (self.dataset_hierarchy
                                .get_classifiable_subtree(synset))
        subtree_indices = [pair.class_id for pair in classifiable_subtree]
        subtree_logits = logits[:, subtree_indices]

        # [N x CLASS]
        subtree_probs = scipy.special.softmax(subtree_logits, axis=1)
        # [N x CLASS] -> [N]
        subtree_marginal_probs = subtree_probs.mean(axis=0)
        # [N] -> []
        subtree_marginal_entropy = scipy.stats.entropy(subtree_marginal_probs)
        return subtree_marginal_entropy


class SubtreeUniformMarginalEntropy(SubtreeMarginalEntropy):
    """Computes subtree marginal entropies for uniform logits."""

    base_name = "subtree_uniform_marginal_entropy"

    def compute_subtree_metric(self, synset_name: str, logits: np.ndarray) -> typing.Any:
        uniform_logits = np.zeros_like(logits)
        return super().compute_subtree_metric(synset_name, uniform_logits)


class SubtreeInProb(SubtreeMetric):
    """Computes total subtree probability."""

    base_name = "subtree_in_prob"

    def compute_subtree_metric(self, synset_name: str,
                               logits: np.ndarray) -> typing.Any:
        synset = self.dataset_hierarchy.get_synset_from_name(synset_name)
        classifiable_subtree = (self.dataset_hierarchy
                                .get_classifiable_subtree(synset))
        subtree_indices = [pair.class_id for pair in classifiable_subtree]

        all_probs = scipy.special.softmax(logits, axis=1)
        subtree_probs = all_probs[:, subtree_indices].sum(axis=1).mean()

        return subtree_probs


class SubtreeUniformInProb(SubtreeInProb):
    base_name = "subtree_uniform_in_prob"

    def compute_subtree_metric(self, synset_name: str,
                               logits: np.ndarray) -> typing.Any:
        uniform_logits = np.zeros_like(logits)
        return super().compute_subtree_metric(synset_name, uniform_logits)


class SubtreeIS(SubtreeMoreThanOneChildMetric):
    """Computes incomplete subtree entropies for the marginal distribution."""

    base_name = "subtree_is"

    def compute_subtree_metric(self, synset_name: str,
                               logits: np.ndarray) -> typing.Any:
        synset = self.dataset_hierarchy.get_synset_from_name(synset_name)
        classifiable_subtree = (self.dataset_hierarchy
                                .get_classifiable_subtree(synset))

        subtree_indices = [pair.class_id for pair in classifiable_subtree]
        subtree_logits = logits[:, subtree_indices]

        # [N x CLASS]
        subtree_probs = scipy.special.softmax(subtree_logits, axis=1)
        # [N x CLASS] -> [CLASS]
        subtree_marginal_probs = subtree_probs.mean(axis=0)
        # [CLASS] -> []
        subtree_kl = scipy.special.rel_entr(
            subtree_probs, subtree_marginal_probs[None, :])
        subtree_kl = subtree_kl.sum(axis=1).mean(axis=0)
        return subtree_kl


class SubtreeUniformIS(SubtreeIS):
    base_name = "subtree_uniform_inception_score"

    def compute_subtree_metric(self, synset_name: str,
                               logits: np.ndarray) -> typing.Any:
        uniform_logits = np.zeros_like(logits)
        return super().compute_subtree_metric(synset_name, uniform_logits)
