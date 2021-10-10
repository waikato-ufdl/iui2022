import os
import sys
from collections import OrderedDict

from tensorflow import keras

from sample import *

INIT_LR = 1e-4
BS = 5
SEED = 42
VALIDATION_PERCENT = 0.15

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])
OUT_PATH = sys.argv[2]
TRAIN_DATASET = sys.argv[3]
VAL_DATASET = sys.argv[4]
NUM_EPOCHS = int(sys.argv[5])
MODEL = sys.argv[6]
PREDICT_DATASETS = sys.argv[7:]

label_indices = label_indices(load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT)))

PREDICTIONS_FILE_HEADER = predictions_file_header(label_indices)

model = model_for_fine_tuning(MODEL, len(label_indices), "imagenet")
opt = keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

train_gen = data_flow_from_disk(SOURCE_PATH, load_dataset(TRAIN_DATASET), label_indices, True, BS, SEED, MODEL)
val_gen = data_flow_from_disk(SOURCE_PATH, load_dataset(VAL_DATASET), label_indices, False, BS, SEED, MODEL)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=NUM_EPOCHS
)

for PREDICT_DATASET in PREDICT_DATASETS:
    path, filename, ext = split_arg(PREDICT_DATASET)
    out_path = os.path.join(OUT_PATH, f"{filename}.predictions" + ext)

    predict_dataset = load_dataset(PREDICT_DATASET)

    predict_gen = data_flow_from_disk(SOURCE_PATH, predict_dataset, label_indices, False, BS, SEED, MODEL)

    predictions: Predictions = OrderedDict()
    for holdout_item, prediction in zip(predict_dataset.keys(), model.predict(predict_gen)):
        predictions[holdout_item] = prediction

    write_predictions(
        predictions,
        PREDICTIONS_FILE_HEADER,
        out_path
    )
