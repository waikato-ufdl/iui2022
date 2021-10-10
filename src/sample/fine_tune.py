import os
import sys
from collections import OrderedDict
from random import Random

from tensorflow import keras

from sample import *
from sample.splitters import RandomSplitter, TopNSplitter

INIT_LR = 1e-4
BS = 5
NUM_EPOCHS = int(sys.argv[2])
SEED = 42
VALIDATION_PERCENT = 0.15

RANDOM = Random(SEED)

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

MODEL = sys.argv[3]

WEIGHTS = sys.argv[4] if len(sys.argv) == 5 else "imagenet"

source_dataset = load_dataset("schedule.txt")

label_indices = label_indices(load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT)))

PREDICTIONS_FILE_HEADER = predictions_file_header(label_indices)

holdout_dataset = load_dataset("holdout.txt")

holdout_gen = data_flow_from_disk(SOURCE_PATH, holdout_dataset, label_indices, False, BS, SEED, MODEL)

splitter = TopNSplitter(50)

iteration = 0
iteration_dataset, remaining_dataset = splitter(source_dataset)
while True:
    print(f"ITERATION {iteration}")

    validation_size = max(int(len(iteration_dataset) * VALIDATION_PERCENT), 1)

    validation_dataset, train_dataset = RandomSplitter(validation_size, RANDOM)(iteration_dataset)

    model = model_for_fine_tuning(MODEL, len(label_indices), WEIGHTS)
    opt = keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

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

    write_predictions(predictions, PREDICTIONS_FILE_HEADER, f"predictions.{MODEL}.{iteration}.txt")

    if len(remaining_dataset) == 0:
        break

    update_dataset, remaining_dataset = splitter(remaining_dataset)

    update_gen = data_flow_from_disk(SOURCE_PATH, update_dataset, label_indices, False, BS, SEED, MODEL)

    predictions: Predictions = OrderedDict()
    for update_item, prediction in zip(update_dataset.keys(), model.predict(update_gen)):
        predictions[update_item] = prediction

    write_predictions(predictions, PREDICTIONS_FILE_HEADER, f"update_predictions.{MODEL}.{iteration}.txt")

    iteration_dataset = merge(iteration_dataset, update_dataset)

    iteration += 1
