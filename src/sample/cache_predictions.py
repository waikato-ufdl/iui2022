import os
import sys

from sample import *


SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

MODEL = sys.argv[2]

source_dataset = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))

header = predictions_file_header(label_indices(source_dataset))

predictions = dataset_predictions(MODEL, SOURCE_PATH, source_dataset)

write_predictions(predictions, header, os.path.join(SOURCE_PATH, f"{SOURCE_DATASET}.{MODEL}-predictions.txt"))
