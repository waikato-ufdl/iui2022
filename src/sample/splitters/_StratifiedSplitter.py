from collections import OrderedDict
from random import Random
from typing import Set

from .._types import Dataset, Split, LabelIndices
from .._util import per_label
from ._RandomSplitter import RandomSplitter
from ._Splitter import Splitter


class StratifiedSplitter(Splitter):
    """
    TODO
    """
    def __init__(self, percentage: float, labels: LabelIndices, random: Random = Random()):
        self._percentage = percentage
        self._labels = labels
        self._random = random

    def __str__(self) -> str:
        return f"strat-{self._percentage}"

    def __call__(self, dataset: Dataset) -> Split:
        subsets_per_label = per_label(dataset)

        sub_splits = {
            label: RandomSplitter(int(len(subsets_per_label[label]) * self._percentage), self._random)(subsets_per_label[label])
            for label in self._labels.keys()
        }

        result = OrderedDict(), OrderedDict()

        for filename, label in dataset.items():
            result_index = 0 if filename in sub_splits[label][0] else 1

            result[result_index][filename] = label

        return result
