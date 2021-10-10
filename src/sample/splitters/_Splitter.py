from abc import ABC, abstractmethod

from .._types import Dataset, Split


class Splitter(ABC):
    """
    TODO
    """
    @abstractmethod
    def __call__(self, dataset: Dataset) -> Split:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
