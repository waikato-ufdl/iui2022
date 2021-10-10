from collections import OrderedDict

from ._Splitter import Splitter
from .. import Dataset, Split


class TopNSplitter(Splitter):
    """
    Splits the first N items of a given dataset.
    """
    def __init__(self, n: int):
        self._n = n

    def __call__(self, dataset: Dataset) -> Split:
        result = OrderedDict(), OrderedDict()

        for index, item in enumerate(dataset.items()):
            filename, label = item
            result[0 if index < self._n else 1][filename] = label

        return result

    def __str__(self) -> str:
        return f"top-{self._n}"
