from typing import OrderedDict, Tuple

import numpy as np

Dataset = OrderedDict[str, str]

Split = Tuple[Dataset, Dataset]

Predictions = OrderedDict[str, np.ndarray]

LabelIndices = OrderedDict[str, int]
