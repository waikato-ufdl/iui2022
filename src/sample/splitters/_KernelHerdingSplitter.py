from collections import OrderedDict
from typing import Union

import numpy as np

from .._kernel import RBFKernel2
from .._load import load_predictions
from .._types import Dataset, Predictions, Split
from .._util import compare_ignore_index
from ._Splitter import Splitter


class KernelHerdingSplitter(Splitter):
    """
    TODO
    """
    def __init__(self, model:str, predictions_path: Union[str, Predictions], count: int):
        self._model = model
        self._predictions = load_predictions(predictions_path) if isinstance(predictions_path, str) else predictions_path
        self._count = count

    def __call__(self, dataset: Dataset) -> Split:
        kernel_list = np.array([self._predictions[filename] for filename in dataset.keys()])
        dataset_size = len(dataset)

        kernel = RBFKernel2(kernel_list)
        print("BUILT KERNEL")

        estimated_expected_similarity = [
            sum(kernel.eval(i, j) for j in range(dataset_size)) / dataset_size
            for i in range(dataset_size)
        ]
        print("ESTIMATED EXPECTED SIMILARITY")

        index, max_score = max(enumerate(estimated_expected_similarity), key=compare_ignore_index)

        num_sampled = 0
        accumulated_similarity_to_sample = [0.0] * dataset_size
        selected = []
        unselected = set(range(dataset_size))
        while True:
            selected.append(index)
            unselected.remove(index)
            num_sampled += 1
            print(f"SELECTED ITEM {index} OF {dataset_size}")
            if num_sampled >= self._count:
                break

            for i in unselected:
                accumulated_similarity_to_sample[i] += kernel.eval(i, index)
            print("UPDATED ACCUMULATED SIMILARITIES")

            index, max_score = max(
                (
                    (i, estimated_expected_similarity[i] - accumulated_similarity_to_sample[i] / (num_sampled + 1))
                    for i in range(dataset_size)
                    if i in unselected
                ),
                key=compare_ignore_index
            )

        index_map = {
            index: filename
            for index, filename in enumerate(dataset.keys())
        }

        selected_set = OrderedDict()
        for index in selected:
            filename = index_map[index]
            selected_set[filename] = dataset[filename]

        unselected_set = OrderedDict()

        for index, filename in enumerate(dataset.keys()):
            if index in unselected:
                unselected_set[filename] = dataset[filename]

        return selected_set, unselected_set

    def __str__(self) -> str:
        return f"kh-{self._model}-{self._count}"
