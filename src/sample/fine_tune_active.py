import os
import sys
from collections import OrderedDict
from random import Random

from tensorflow import keras

from sample import *
from sample.splitters import RandomSplitter, RankedEntropySplitter, SoftmaxBALDSplitter2, KernelHerdingSplitter

INIT_LR = 1e-4
BS = 5
SEED = 42
VALIDATION_PERCENT = 0.15

RANDOM = Random(SEED)

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])
NUM_EPOCHS = int(sys.argv[2])
MODEL = sys.argv[3]
if sys.argv[4] == "ranked":
    SPLITTER_FACTORY = RankedEntropySplitter
    SPLITTER = "ranked"
elif sys.argv[4] == "softmaxBALD":
    TEMPERATURE = float(sys.argv[5])
    SPLITTER = f"softmaxBALD-{TEMPERATURE}"

    def SPLITTER_FACTORY(pred, size):
        return SoftmaxBALDSplitter2(pred, size, TEMPERATURE, RANDOM)
elif sys.argv[4] == "kh":
    SPLITTER = "kh"

    def SPLITTER_FACTORY(pred, size):
        return KernelHerdingSplitter(MODEL, pred, size)
else:
    raise Exception(f"Unknown splitter '{sys.argv[4]}'")

source_dataset = load_dataset("schedule.txt")

label_indices = label_indices(load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT)))

PREDICTIONS_FILE_HEADER = predictions_file_header(label_indices)

holdout_dataset = load_dataset("holdout.txt")

holdout_gen = data_flow_from_disk(SOURCE_PATH, holdout_dataset, label_indices, False, BS, SEED, MODEL)


def split_dataset(dataset: Dataset, model: keras.models.Model) -> Split:
    dataset_gen = data_flow_from_disk(SOURCE_PATH, dataset, label_indices, False, BS, SEED, MODEL)
    predictions: Predictions = OrderedDict()
    for dataset_item, prediction in zip(dataset.keys(), model.predict(dataset_gen)):
        predictions[dataset_item] = prediction
    return SPLITTER_FACTORY(predictions, 50)(dataset)


iteration = 0
while True:
    print(f"ITERATION {iteration}")

    model = model_for_fine_tuning(MODEL, len(label_indices), "imagenet")
    opt = keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    if iteration == 0:
        iteration_dataset, remaining_dataset = split_dataset(source_dataset, model)

    validation_size = max(int(len(iteration_dataset) * VALIDATION_PERCENT), 1)

    validation_dataset, train_dataset = RandomSplitter(validation_size, RANDOM)(iteration_dataset)

    train_gen = data_flow_from_disk(SOURCE_PATH, train_dataset, label_indices, True, BS, SEED, MODEL)
    val_gen = data_flow_from_disk(SOURCE_PATH, validation_dataset, label_indices, False, BS, SEED, MODEL)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS
    )

    predictions: Predictions = OrderedDict()
    for holdout_item, prediction in zip(holdout_dataset.keys(), model.predict(holdout_gen)):
        predictions[holdout_item] = prediction

    write_predictions(predictions, PREDICTIONS_FILE_HEADER, f"predictions.{MODEL}.{SPLITTER}.{iteration}.txt")

    if len(remaining_dataset) == 0:
        break

    update_dataset, remaining_dataset = split_dataset(remaining_dataset, model)

    update_gen = data_flow_from_disk(SOURCE_PATH, update_dataset, label_indices, False, BS, SEED, MODEL)

    predictions: Predictions = OrderedDict()
    for update_item, prediction in zip(update_dataset.keys(), model.predict(update_gen)):
        predictions[update_item] = prediction

    write_predictions(predictions, PREDICTIONS_FILE_HEADER, f"update_predictions.{MODEL}.{SPLITTER}.{iteration}.txt")

    iteration_dataset = merge(iteration_dataset, update_dataset)

    iteration += 1
