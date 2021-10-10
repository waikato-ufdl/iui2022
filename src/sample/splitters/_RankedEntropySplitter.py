from collections import OrderedDict

from ..scoring import entropy
from .._types import Dataset, Predictions, Split
from ._Splitter import Splitter


class RankedEntropySplitter(Splitter):
    """
    TODO
    """
    def __init__(
            self,
            predictions: Predictions,
            batch_size: int
    ):
        self._predictions = predictions
        self._batch_size = batch_size

    def __call__(self, dataset: Dataset) -> Split:
        # Calculate the entropy for each item
        scores = [
            (filename, entropy(self._predictions[filename]))
            for filename in dataset
        ]

        scores.sort(key=lambda x: x[1], reverse=True)

        result = OrderedDict(), OrderedDict()

        for filename, _ in scores:
            result_index = 0 if len(result[0]) < self._batch_size else 1

            result[result_index][filename] = dataset[filename]

        return result

    def __str__(self) -> str:
        return f"ranked-entropy"
