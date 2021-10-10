from collections import OrderedDict
from random import Random
from typing import Set

from .._types import Dataset, Split
from .._util import per_label
from ._RandomSplitter import RandomSplitter
from ._Splitter import Splitter


class UniformSplitter(Splitter):
    """
    TODO
    """
    def __init__(self, count_per_label: int, labels: Set[str], random: Random = Random()):
        self._count_per_label = count_per_label
        self._labels = labels
        self._random = random

    def __str__(self) -> str:
        return f"uni-{self._count_per_label}"

    def __call__(self, dataset: Dataset) -> Split:
        subsets_per_label = per_label(dataset)
        sub_splitter = RandomSplitter(self._count_per_label, self._random)
        sub_splits = {
            label: sub_splitter(subsets_per_label[label])
            for label in self._labels
        }

        result = OrderedDict(), OrderedDict()

        for filename, label in dataset.items():
            result_index = 0 if filename in sub_splits[label][0] else 1

            result[result_index][filename] = label

        return result
