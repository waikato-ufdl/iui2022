import os
from random import Random
import sys

from sample import *
from sample.splitters import *
from sample.schedulers import *

RANDOM = Random(42)
HOLDOUT_PERCENT = 0.15

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])
MODEL = sys.argv[2]

print(f"SOURCE PATH = {SOURCE_PATH}")
print(f"SOURCE DATASET = {SOURCE_DATASET}{SOURCE_EXT}")

print(f"HOLDOUT PERCENTAGE = {HOLDOUT_PERCENT}")

source_dataset = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))

print(f"NUM ITEMS = {len(source_dataset)}")
for label, label_dataset in per_label(source_dataset).items():
    print(f"  {label}: {len(label_dataset)}")

LABELS = label_indices(source_dataset)
NUM_LABELS = len(LABELS)

print(f"NUM LABELS = {NUM_LABELS}")

HOLDOUT_SPLITTER: Splitter = StratifiedSplitter(HOLDOUT_PERCENT, LABELS, RANDOM)

TRAIN_SCHEDULER: Scheduler = RandomScheduler(RANDOM)
#TRAIN_SCHEDULER: Scheduler = UniformScheduler(RANDOM)
#TRAIN_SCHEDULER: Scheduler = StratifiedScheduler(RANDOM)
#TRAIN_SCHEDULER: Scheduler = KernelHerdingScheduler(MODEL, os.path.join(SOURCE_PATH, f"{SOURCE_DATASET}.{MODEL}-predictions.txt"))

DEST_PATH = os.path.join(SOURCE_PATH, f"{SOURCE_DATASET}.{HOLDOUT_SPLITTER}.{TRAIN_SCHEDULER}.splits")
os.makedirs(DEST_PATH, exist_ok=True)

print(f"DEST PATH = {DEST_PATH}")

holdout_dataset, left_in_dataset = HOLDOUT_SPLITTER(source_dataset)
holdout_dataset_dest = os.path.join(DEST_PATH, "holdout" + SOURCE_EXT)
write_dataset(holdout_dataset, holdout_dataset_dest)
print(f"WROTE HOLDOUT DATASET TO {holdout_dataset_dest}")

schedule = TRAIN_SCHEDULER(left_in_dataset)
with open(os.path.join(DEST_PATH, "schedule.txt"), "w") as file:
    for item in schedule:
        file.write(item + "\n")
