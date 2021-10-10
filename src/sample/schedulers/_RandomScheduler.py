from random import Random
from typing import List

from ._Scheduler import Scheduler
from .._math import random_permutation
from .._types import Dataset


class RandomScheduler(Scheduler):
    """
    TODO
    """
    def __init__(self, random: Random = Random()):
        self._random = random

    def __call__(self, dataset: Dataset) -> List[str]:
        return random_permutation(list(dataset.keys()), self._random)

    def __str__(self) -> str:
        return "rand"
