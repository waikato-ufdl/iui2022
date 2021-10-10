import os.path
from collections import OrderedDict
from typing import Tuple, OrderedDict as ODict

import numpy as np

from ._types import Dataset, Predictions


def load_dataset(filename: str) -> Dataset:
    """
    TODO
    """
    result = OrderedDict()

    with open(filename, "r") as file:
        while True:
            item = file.readline().rstrip()
            if item == "":
                break
            label = os.path.split(os.path.split(item)[0])[1]
            result[item] = label

    return result


def load_predictions(filename: str) -> Predictions:
    """
    TODO
    """
    result = OrderedDict()

    with open(filename, "r") as file:
        file.readline()
        while True:
            item = file.readline().rstrip()
            if item == "":
                break
            split = item.split(",")
            filename = split[0]
            probs = [float(prob) for prob in split[1:]]
            result[filename] = np.array(probs)

    return result


def load_rois_predictions(dir: str, dataset: Dataset, num_labels: int) -> Predictions:
    """
    TODO
    """
    result = OrderedDict()
    for filename in dataset.keys():
        probabilities = np.zeros((num_labels,))
        rois_filename = os.path.basename(filename)[:-4] + "-rois.csv"
        with open(os.path.join(dir, rois_filename), "r") as file:
            file.readline()
            for line in file.readlines():
                label, _, score = line.strip().split(",")[-3:]
                index = int(label)
                probabilities[index] = max(probabilities[index], float(score))
        result[filename] = probabilities
    return result


def get_highest_score_bbox(
        dir: str,
        dataset: Dataset
) -> ODict[str, Tuple[int, int, int, int]]:
    """
    TODO
    """
    result = OrderedDict()
    for filename in dataset.keys():
        best_bbox = None
        best_score = None
        rois_filename = os.path.basename(filename)[:-4] + "-rois.csv"
        with open(os.path.join(dir, rois_filename), "r") as file:
            file.readline()
            for line in file.readlines():
                _, x0, y0, x1, y1, _, _, _, _, _, _, score = line.strip().split(",")
                score = float(score)
                if best_score is None or best_score < score:
                    best_score = score
                    best_bbox = (
                        round(float(x0)),
                        round(float(y0)),
                        round(float(x1)),
                        round(float(y1))
                    )
        if best_bbox is not None:
            result[filename] = best_bbox
    return result
