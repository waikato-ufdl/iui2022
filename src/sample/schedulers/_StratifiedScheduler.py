from random import Random
from typing import List

from ._Scheduler import Scheduler
from .._math import calculate_schedule
from .._types import Dataset
from .._util import per_label


class StratifiedScheduler(Scheduler):
    """
    TODO
    """
    def __init__(self, random: Random = Random()):
        self._random = random

    def __call__(self, dataset: Dataset) -> List[str]:
        remaining = list(list(label_set.keys()) for label_set in per_label(dataset).values())

        ratios = tuple(map(len, remaining))

        schedule = calculate_schedule(ratios)

        result = []

        i=0
        for index in schedule:
            print(i)
            i+=1
            label_set = remaining[index]
            selected_index = self._random.randrange(len(label_set))
            result.append(label_set[selected_index])
            label_set.pop(selected_index)

        return result

    def __str__(self) -> str:
        return "strat"
