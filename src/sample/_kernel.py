import math
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class RBFKernel2:
    """
    TODO
    """
    def __init__(self, data: np.ndarray):
        self._data = data
        self._gamma: float = 0.01
        self._kernel_precalc = np.sum(np.square(data), axis=1)
        data_range = np.arange(data.shape[0])
        precalcs_id1 = self._kernel_precalc[data_range, np.newaxis]
        precalcs_id2 = self._kernel_precalc[np.newaxis, data_range]
        dps = np.tensordot(
            data,
            data,
            axes=([1], [1])
        )
        self._kernel_matrix = np.exp(-self._gamma * (precalcs_id1 - 2 * dps + precalcs_id2))

    def eval(self, id1: int, id2: int) -> float:
        return self._kernel_matrix[id1][id2]


class Kernel(ABC):
    """
    TODO
    """
    def __init__(self):
        self._data: Optional[List[np.ndarray]] = None

    def build_kernel(self, data: List[np.ndarray]):
        self.init_vars(data)

    def init_vars(self, data: List[np.ndarray]):
        self._data = data

    @abstractmethod
    def eval(self, id1: int, id2: int, inst1: np.ndarray) -> float:
        pass


class CachedKernel(Kernel, ABC):
    """
    TODO
    """
    def __init__(self):
        super().__init__()
        self._kernel_evals: int = 0
        self._cache_hits: int = 0
        self._cache_size: int = 250007
        self._storage: Optional[List[float]] = None
        self._keys: Optional[List[int]] = None
        self._kernel_matrix: Optional[List[List[float]]] = None
        self._num_insts: int = 0
        self._cache_slots: int = 4

    def init_vars(self, data: List[np.ndarray]):
        super().init_vars(data)

        self._kernel_evals = 0
        self._cache_hits = 0
        self._num_insts = len(data)

        if self._cache_size > 0:
            self._storage = [0.0] * (self._cache_size * self._cache_slots)
            self._keys = [0] * (self._cache_size * self._cache_slots)
        else:
            self._storage = self._keys = self._kernel_matrix = None

    @abstractmethod
    def evaluate(self, id1: int, id2: int, inst1: np.ndarray) -> float:
        pass

    def eval(self, id1: int, id2: int, inst1: np.ndarray) -> float:
        result: float = 0.0
        key: int = -1
        location: int = -1

        if id1 >= 0 and self._cache_size != -1:
            if self._cache_size == 0:
                if self._kernel_matrix is None:
                    self._kernel_matrix = []
                    for i in range(len(self._data)):
                        self._kernel_matrix.append([])
                        for j in range(i + 1):
                            self._kernel_evals += 1
                            self._kernel_matrix[i].append(self.evaluate(i, j, self._data[i]))
                self._cache_hits += 1
                return self._kernel_matrix[id1][id2] if id1 > id2 else self._kernel_matrix[id2][id1]

            key = id1 + id2 * self._num_insts if id1 > id2 else id2 + id1 * self._num_insts
            location = (key % self._cache_size) * self._cache_slots
            loc = location
            for i in range(self._cache_slots):
                this_key = self._keys[loc]
                if this_key == 0:
                    break
                if this_key == key + 1:
                    self._cache_hits += 1
                    if i > 0:
                        tmps = self._storage[loc]
                        self._storage[loc] = self._storage[location]
                        self._keys[loc] = self._keys[location]
                        self._storage[location] = tmps
                        self._keys[location] = this_key
                        return tmps
                    else:
                        return self._storage[loc]
                loc += 1

        result = self.evaluate(id1, id2, inst1)

        self._kernel_evals += 1

        if key != -1 and self._cache_size != -1:
            for i in reversed(range(location, location + self._cache_slots - 1)):
                self._keys[i + 1] = self._keys[i]
                self._storage[i + 1] = self._storage[i]
            self._storage[location] = result
            self._keys[location] = key + 1

        return result

    def dot_product(self, inst1: np.ndarray, inst2: np.ndarray) -> float:
        return float(np.vdot(inst1, inst2))


class RBFKernel(CachedKernel):
    """
    TODO
    """
    def __init__(self):
        super().__init__()
        self._kernel_precalc = None
        self._gamma: float = 0.01

    def build_kernel(self, data: List[np.ndarray]):
        super().build_kernel(data)

        self._kernel_precalc = [
            np.sum(np.square(item))
            for item in data
        ]

    def evaluate(self, id1: int, id2: int, inst1: np.ndarray) -> float:
        if id1 == id2:
            return 1.0
        else:
            if id1 == -1:
                return math.exp(
                    -self._gamma * (
                        self.dot_product(inst1, inst1) - 2 * self.dot_product(inst1, self._data[id2])
                        + self._kernel_precalc[id2]
                    )
                )
            else:
                return math.exp(
                    -self._gamma * (
                            self._kernel_precalc[id1] - 2 * self.dot_product(inst1, self._data[id2])
                            + self._kernel_precalc[id2]
                    )
                )
