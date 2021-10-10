from collections import OrderedDict
import os
from random import Random
import subprocess
from typing import Iterable, List, Optional, Tuple, OrderedDict as ODict

import numpy as np

from ._math import random_permutation
from ._types import Dataset, LabelIndices


def per_label(dataset: Dataset) -> ODict[str, Dataset]:
    """
    TODO
    """
    result = OrderedDict()

    for filename, label in dataset.items():
        if label in result:
            subset = result[label]
        else:
            subset = OrderedDict()
            result[label] = subset

        subset[filename] = label

    return result


def merge(d1: Dataset, d2: Dataset) -> Dataset:
    """
    TODO
    """
    return OrderedDict(**d1, **d2)


def split_arg(arg: str) -> Tuple[str, str, str]:
    """
    TODO
    """
    path_split = os.path.split(arg)
    return (path_split[0], *os.path.splitext(path_split[1]))


def first(iterable: Iterable):
    for el in iterable:
        return el


def compare_ignore_index(item: Tuple[int, float]) -> float:
    return item[1]


def label_indices(dataset: Dataset) -> LabelIndices:
    """
    TODO
    """
    result = OrderedDict()
    for label in dataset.values():
        if label not in result:
            result[label] = len(result)
    return result


def predictions_file_header(indices: LabelIndices) -> str:
    """
    TODO
    """
    return f"filename,{','.join(f'{label}_prob' for label in indices.keys())}\n"


def top_n(dataset: Dataset, n: int) -> Dataset:
    """
    TODO
    """
    result = OrderedDict()
    for i, filename in enumerate(dataset.keys()):
        if i >= n:
            break
        result[filename] = dataset[filename]
    return result


def change_path(
        dataset: Dataset,
        path: str,
        subdir: bool = False,
        ext: Optional[str] = "xml"
) -> Dataset:
    """
    TODO
    """
    result = OrderedDict()
    for filename, label in dataset.items():
        changed_filename = change_filename(filename, path, label if subdir else None, ext)
        result[changed_filename] = dataset[filename]
    return result


def change_filename(filename: str, path: str, label: Optional[str] = None, ext: Optional[str] = "xml") -> str:
    """
    TODO
    """
    _, file = os.path.split(filename)
    if ext is not None:
        file, _ = os.path.splitext(file)
        file = f"{file}.{ext}"
    if label is not None:
        path = os.path.join(path, label)
    return os.path.join(path, file)


def shuffle_dataset(dataset: Dataset, random: Random = Random()) -> Dataset:
    """
    TODO
    """
    order = random_permutation(list(dataset.keys()), random)
    result = OrderedDict()
    for filename in order:
        result[filename] = dataset[filename]
    return result


def rm_dir(path: str):
    """
    TODO
    """
    for dirpath, dirnames, filenames in os.walk(path, False, followlinks=False):
        for filename in filenames:
            os.remove(os.path.join(dirpath, filename))
        for dirname in dirnames:
            os.rmdir(os.path.join(dirpath, dirname))
    os.rmdir(path)


def run_command(cmd: str):
    """
    TODO
    """
    with subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE) as process:
        try:
            process.wait()
        except KeyboardInterrupt:
            process.kill()
            raise


def coerce_incorrect(num_labels: int, y_true: int, y_score: np.ndarray) -> np.ndarray:
    """
    TODO
    """
    if not np.all(y_score == 0.0):
        return y_score

    y_false = (y_true + 1) % num_labels

    result = y_score.copy()
    result[y_false] = 1.0
    return result


def set_all_labels(
        dataset: Dataset,
        label: str
) -> Dataset:
    """
    TODO
    """
    result = OrderedDict()
    for filename in dataset:
        result[filename] = label
    return result
