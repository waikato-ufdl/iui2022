from collections import OrderedDict
from random import Random

from .._math import subset_number_to_subset, number_of_subsets
from .._types import Dataset, Split
from ._Splitter import Splitter


class RandomSplitter(Splitter):
    """
    TODO
    """
    def __init__(self, count: int, random: Random = Random()):
        self._count = count
        self._random = random

    def __str__(self) -> str:
        return f"rand-{self._count}"

    def __call__(self, dataset: Dataset) -> Split:
        dataset_size = len(dataset)

        if self._count > dataset_size:
            raise ValueError(f"Can't select sub-set of size {self._count} from data-set of size {dataset_size}")

        choice_set = subset_number_to_subset(
            dataset_size,
            self._count,
            self._random.randrange(number_of_subsets(dataset_size, self._count))
        )

        result = OrderedDict(), OrderedDict()

        for index, filename in enumerate(dataset.keys()):
            result_index = 0 if index in choice_set else 1

            result[result_index][filename] = dataset[filename]

        return result
