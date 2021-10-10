from collections import OrderedDict
from math import exp
from random import Random

from ..scoring import entropy
from .._types import Dataset, Split, Predictions
from ._Splitter import Splitter


class SoftmaxBALDSplitter2(Splitter):
    """
    TODO
    """
    def __init__(
            self,
            predictions: Predictions,
            batch_size: int,
            temperature: float,
            rand: Random
    ):
        self._predictions = predictions
        self._batch_size = batch_size
        self._temperature = temperature
        self._rand = rand

    def __call__(self, dataset: Dataset) -> Split:
        # Calculate the entropy for each item
        scores = [
            [filename, exp(entropy(self._predictions[filename]) / self._temperature)]
            for filename in dataset
        ]

        result = OrderedDict(), OrderedDict()

        normalisation_factor = sum(score for _, score in scores)

        while len(result[0]) < min(self._batch_size, len(scores)):
            selection = self._rand.random() * normalisation_factor
            for filename, score in scores:
                if filename in result[0]:
                    continue
                selection -= score
                if selection < 0.0:
                    result[0][filename] = dataset[filename]
                    normalisation_factor -= score
                    break

        for filename in dataset:
            if filename not in result[0]:
                result[1][filename] = dataset[filename]

        return result

    def __str__(self) -> str:
        return f"softmaxBALD-{self._temperature}"
