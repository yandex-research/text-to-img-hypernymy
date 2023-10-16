"""Dataset hierarchy."""

import dataclasses
import functools
import typing
import nltk
from nltk.corpus import wordnet
from src import utils


Synset = nltk.corpus.reader.wordnet.Synset


@dataclasses.dataclass(frozen=True)
class ClassSynsetPair:
    """Class id and synset pair."""
    class_id: int
    synset: Synset


class Hierarchy:
    """A class for traversing the dataset hierarchy."""

    def __init__(self, wordnet_id_list_path: str):
        nltk.download("wordnet")
        nltk.download("omw-1.4")

        with open(wordnet_id_list_path, "r", encoding="utf-8") as file:
            wordnet_ids_list = file.read().splitlines()

        self.dataset_class_synsets = set()
        self.class_synset_pairs = []
        self.synset_to_class_pair = {}

        for class_id, wnid in enumerate(wordnet_ids_list):
            synset = wordnet.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))
            self.dataset_class_synsets.add(synset)
            self.class_synset_pairs.append(ClassSynsetPair(class_id, synset))
            self.synset_to_class_pair[synset] = ClassSynsetPair(class_id,
                                                                synset)

    def _filter_tree(self, tree: list, remove_leaves: bool
                     ) -> typing.Tuple[typing.List, bool]:
        """Removes nodes that aren't in the dataset hierarchy from wordnet."""
        node, *subtrees = tree

        if subtrees:
            subtree_filter_results = [
                self._filter_tree(subtree, remove_leaves)
                for subtree in subtrees]
            filtered_subtrees, is_in_hierarchy = zip(*subtree_filter_results)
        else:
            filtered_subtrees = []
            is_in_hierarchy = []

        if any(is_in_hierarchy):
            # If some subtree is in the hierarchy we add the current node.
            non_empty_subtrees = [
                subtree for subtree in filtered_subtrees if subtree]
            return [node, non_empty_subtrees], True

        if node in self.dataset_class_synsets:
            # If the current node is a dataset class.
            if not remove_leaves:
                # If not remove_leaves, we add it to the final list.
                return [node, []], True
            # Otherwise we don't add it to the list and mark it True.
            return [], True

        # If the node has no subtrees in the hierarchy and isn't a leaf
        # node itself, we don't add and return False.
        return [], False

    @functools.lru_cache
    def get_all_synsets(self, remove_leaves: bool) -> typing.List[Synset]:
        """Returns all the synsets in the hierarchy."""
        tree = wordnet.synset("entity.n.01").tree(lambda s: s.hyponyms())
        filtered_tree, _ = self._filter_tree(tree, remove_leaves)

        all_synsets = utils.unique_list(utils.flatten_list(filtered_tree))
        return all_synsets

    def get_synset_subtree(self, parent_synset: Synset,
                           remove_leaves: bool) -> typing.List[Synset]:
        """Returns the synset subtree of a given node."""
        child_synsets = utils.unique_list(utils.flatten_list(
            parent_synset.tree(lambda s: s.hyponyms())))

        subtree = [child for child in child_synsets
                   if child in self.get_all_synsets(remove_leaves)]
        return subtree

    def get_classifiable_subtree(self, parent_synset: Synset,
                                 ) -> typing.List[ClassSynsetPair]:
        """Returns all children of a node in the hierarchy."""

        child_synsets = utils.unique_list(utils.flatten_list(
            parent_synset.tree(lambda s: s.hyponyms())))

        hierarchy_child_synsets = [child for child in child_synsets
                                   if child in self.dataset_class_synsets]

        child_classes = [self.synset_to_class_pair[synset]
                         for synset in hierarchy_child_synsets]
        return child_classes

    def get_synset_from_name(self, synset_name: str):
        """Returns a synset from a name (e.g. 'dog.n.01')."""
        return wordnet.synset(synset_name)
