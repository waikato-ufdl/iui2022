from collections import OrderedDict
from math import exp

from ..scoring import entropy
from .._types import Dataset, Split, Predictions
from ._Splitter import Splitter


class SoftmaxBALDSplitter(Splitter):
    """
    TODO
    """
    def __init__(
            self,
            predictions: Predictions,
            batch_size: int,
            temperature: float
    ):
        self._predictions = predictions
        self._batch_size = batch_size
        self._temperature = temperature

    def __call__(self, dataset: Dataset) -> Split:
        # Calculate the entropy for each item
        scores = [
            [filename, entropy(self._predictions[filename])]
            for filename in dataset
        ]

        result = OrderedDict(), OrderedDict()

        while True:
            max_index: int = max(enumerate(scores), key=lambda x: x[1][1])[0]
            filename = scores.pop(max_index)[0]
            result[0][filename] = dataset[filename]
            if len(result[0]) == self._batch_size or len(scores) == 0:
                break
            score_deltas_unnormalised = [exp(score / self._temperature) for _, score in scores]
            normalisation_factor = sum(score_deltas_unnormalised)
            for i in range(len(scores)):
                scores[i][1] -= score_deltas_unnormalised[i] / normalisation_factor

        for filename, _ in scores:
            result[1][filename] = dataset[filename]

        return result

    def __str__(self) -> str:
        return f"softmaxBALD-{self._temperature}"
