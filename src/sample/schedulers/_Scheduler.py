from abc import ABC, abstractmethod
from typing import List

from .._types import Dataset


class Scheduler(ABC):
    """
    TODO
    """
    @abstractmethod
    def __call__(self, dataset: Dataset) -> List[str]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
