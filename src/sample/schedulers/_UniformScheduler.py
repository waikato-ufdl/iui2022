from collections import OrderedDict
from random import Random
from typing import List

from ._Scheduler import Scheduler
from .._math import random_permutation
from .._types import Dataset
from .._util import per_label


class UniformScheduler(Scheduler):
    """
    TODO
    """
    def __init__(self, random: Random = Random()):
        self._random = random

    def __call__(self, dataset: Dataset) -> List[str]:
        unselected = OrderedDict(
            (label, list(label_set.keys())) for label, label_set in per_label(dataset).items()
        )

        result = []

        while len(unselected) > 0:
            to_remove = set()

            unselected_labels = list(unselected.keys())
            unselected_labels = random_permutation(unselected_labels, self._random)

            for label in unselected_labels:
                label_set = unselected[label]
                selectable = len(label_set)
                selected_index = self._random.randrange(selectable)
                result.append(label_set[selected_index])
                label_set.pop(selected_index)
                if selectable == 1:
                    to_remove.add(label)
            for label in to_remove:
                unselected.pop(label)

        return result

    def __str__(self) -> str:
        return "uni"
