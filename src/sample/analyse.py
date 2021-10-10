import os

from sklearn.metrics import top_k_accuracy_score
from typing.io import TextIO

from sample import load_dataset, load_predictions, label_indices as get_label_indices, load_rois_predictions, coerce_incorrect
from sample.splitters import TopNSplitter

ic_models = [
    "mobilenet",
    "resnet50",
    "resnet152"
]
od_models = [
    #"r101"
]
hybrid_models = [
    #"r101-resnet50",
    #"r101-resnet50-predicted"
]
active_models = [
    "mobilenet.ranked",
    "mobilenet.softmaxBALD-0.125",
    "mobilenet.kh",
    "resnet50.ranked",
    "resnet50.softmaxBALD-0.125",
    "resnet50.kh",
    "resnet152.ranked",
    "resnet152.softmaxBALD-0.125",
    "resnet152.kh"
]

datasets = [
    "asl",
    "flowers-files",
    "dog_breeds",
    #"dog_breeds_4", "dog_breeds_8", "dog_breeds_16", "dog_breeds_32",
    #"dog_breeds_top_half", "dog_breeds_top_half_4", "dog_breeds_top_half_8", "dog_breeds_top_half_16",
    "birds"
]
splits = [
    "rand",
    "uni",
    "kh"
]


def analyse_od(model: str, dataset: str, split: str, pivot_file: TextIO):
    """
    TODO
    """
    if split == "kh":
        return

    source_dataset = load_dataset(f"{dataset}.txt")

    label_indices = get_label_indices(source_dataset)

    numeric_labels = list(range(len(label_indices)))
    num_labels = len(numeric_labels)

    split_name = split if split != "kh" else f"kh-{model}"
    split_path = f"{dataset}.strat-0.15.{split_name}.splits"

    holdout_dataset = load_dataset(os.path.join(split_path, "holdout.txt"))

    schedule_dataset = load_dataset(os.path.join(split_path, "schedule.txt"))

    y_true = [label_indices[label] for label in holdout_dataset.values()]

    splitter = TopNSplitter(50)

    iteration = 0
    cumulative_corrections = 0
    _, remaining_dataset = splitter(schedule_dataset)
    while True:
        holdout_predictions_path = os.path.join(split_path, f"{model}/{iteration}/predictions")

        if not os.path.exists(holdout_predictions_path):
            break

        holdout_predictions = load_rois_predictions(holdout_predictions_path, holdout_dataset, num_labels)

        y_score = list(holdout_predictions.values())

        y_score = [coerce_incorrect(num_labels, truth, prediction)
                   for truth, prediction in zip(y_true, y_score)]

        top_1 = top_k_accuracy_score(
            y_true,
            y_score,
            k=1,
            labels=numeric_labels,
            normalize=True
        )

        pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "holdout", "accuracy", top_1])) + "\n")

        update_dataset, remaining_dataset = splitter(remaining_dataset)

        update_predictions_path = os.path.join(split_path, f"{model}/{iteration}/update_predictions")

        if os.path.exists(update_predictions_path):
            update_y_true = [label_indices[label] for label in update_dataset.values()]
            update_predictions = load_rois_predictions(update_predictions_path, update_dataset, num_labels)
            update_y_score = list(update_predictions.values())
            update_y_score = [coerce_incorrect(num_labels, truth, prediction) for truth, prediction in zip(update_y_true, update_y_score)]
            update_top_1 = top_k_accuracy_score(update_y_true, update_y_score, k=1, labels=numeric_labels, normalize=True)
            pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "update", "accuracy", update_top_1])) + "\n")
            cumulative_corrections += int((1 - update_top_1) * 50)
            pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "update", "cumulative_corrections", cumulative_corrections])) + "\n")

        iteration += 1


def analyse_ic(model: str, dataset: str, split: str, pivot_file: TextIO):
    """
    TODO
    """
    source_dataset = load_dataset(f"{dataset}.txt")

    label_indices = get_label_indices(source_dataset)

    numeric_labels = list(range(len(label_indices)))

    split_name = split if split != "kh" else f"kh-{model}"
    split_path = f"{dataset}.strat-0.15.{split_name}.splits"

    holdout_dataset = load_dataset(os.path.join(split_path, "holdout.txt"))

    schedule_dataset = load_dataset(os.path.join(split_path, "schedule.txt"))

    y_true = [label_indices[label] for label in holdout_dataset.values()]

    splitter = TopNSplitter(50)

    iteration = 0
    cumulative_corrections = 0
    _, remaining_dataset = splitter(schedule_dataset)
    while True:
        holdout_predictions_filename = os.path.join(split_path, f"predictions.{model}.{iteration}.txt")

        if not os.path.exists(holdout_predictions_filename):
            break

        holdout_predictions = load_predictions(holdout_predictions_filename)

        y_score = list(holdout_predictions.values())

        top_1 = top_k_accuracy_score(
            y_true,
            y_score,
            k=1,
            labels=numeric_labels
        )

        pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "holdout", "accuracy", top_1])) + "\n")

        update_dataset, remaining_dataset = splitter(remaining_dataset)

        update_predictions_filename = os.path.join(split_path, f"update_predictions.{model}.{iteration}.txt")

        if os.path.exists(update_predictions_filename):
            update_y_true = [label_indices[label] for label in update_dataset.values()]
            update_predictions = load_predictions(update_predictions_filename)
            update_y_score = list(update_predictions.values())
            update_top_1 = top_k_accuracy_score(update_y_true, update_y_score, k=1, labels=numeric_labels)
            pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "update", "accuracy", update_top_1])) + "\n")
            cumulative_corrections += int((1 - update_top_1) * 50)
            pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "update", "cumulative_corrections", cumulative_corrections])) + "\n")

        iteration += 1


def analyse_hybrid(model: str, dataset: str, split: str, pivot_file: TextIO):
    """
    TODO
    """
    source_dataset = load_dataset(f"{dataset}.txt")

    label_indices = get_label_indices(source_dataset)

    numeric_labels = list(range(len(label_indices)))

    split_name = split if split != "kh" else f"kh-{model}"
    split_path = f"{dataset}.strat-0.15.{split_name}.splits"

    holdout_dataset = load_dataset(os.path.join(split_path, "holdout.txt"))

    schedule_dataset = load_dataset(os.path.join(split_path, "schedule.txt"))

    y_true = [label_indices[label] for label in holdout_dataset.values()]

    splitter = TopNSplitter(50)

    iteration = 0
    cumulative_corrections = 0
    _, remaining_dataset = splitter(schedule_dataset)
    while True:
        holdout_predictions_filename = os.path.join(
            split_path,
            model,
            str(iteration),
            "holdout_cropped.predictions.txt"
        )

        if not os.path.exists(holdout_predictions_filename):
            break

        holdout_predictions = load_predictions(holdout_predictions_filename)

        y_score = list(holdout_predictions.values())

        top_1 = top_k_accuracy_score(
            y_true,
            y_score,
            k=1,
            labels=numeric_labels
        )

        pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "holdout", "accuracy", top_1])) + "\n")

        update_dataset, remaining_dataset = splitter(remaining_dataset)

        update_predictions_filename = os.path.join(
            split_path,
            model,
            str(iteration),
            "update_cropped.predictions.txt"
        )

        if os.path.exists(update_predictions_filename):
            update_y_true = [label_indices[label] for label in update_dataset.values()]
            update_predictions = load_predictions(update_predictions_filename)
            update_y_score = list(update_predictions.values())
            update_top_1 = top_k_accuracy_score(update_y_true, update_y_score, k=1, labels=numeric_labels)
            pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "update", "accuracy", update_top_1])) + "\n")
            cumulative_corrections += int((1 - update_top_1) * 50)
            pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "update", "cumulative_corrections", cumulative_corrections])) + "\n")

        iteration += 1


def analyse_active(model: str, dataset: str, split: str, pivot_file: TextIO):
    """
    TODO
    """
    if split != "uni":
        return

    source_dataset = load_dataset(f"{dataset}.txt")

    label_indices = get_label_indices(source_dataset)

    numeric_labels = list(range(len(label_indices)))

    split_path = f"{dataset}.strat-0.15.uni.splits"

    holdout_dataset = load_dataset(os.path.join(split_path, "holdout.txt"))

    y_true = [label_indices[label] for label in holdout_dataset.values()]

    iteration = 0
    cumulative_corrections = 0
    while True:
        holdout_predictions_filename = os.path.join(split_path, f"predictions.{model}.{iteration}.txt")

        if not os.path.exists(holdout_predictions_filename):
            break

        holdout_predictions = load_predictions(holdout_predictions_filename)

        y_score = list(holdout_predictions.values())

        top_1 = top_k_accuracy_score(
            y_true,
            y_score,
            k=1,
            labels=numeric_labels
        )

        pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "holdout", "accuracy", top_1])) + "\n")

        update_predictions_filename = os.path.join(split_path, f"update_predictions.{model}.{iteration}.txt")

        if os.path.exists(update_predictions_filename):
            update_predictions = load_predictions(update_predictions_filename)
            update_y_true = [label_indices[source_dataset[filename]] for filename in update_predictions]
            update_y_score = list(update_predictions.values())
            update_top_1 = top_k_accuracy_score(update_y_true, update_y_score, k=1, labels=numeric_labels)
            pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "update", "accuracy", update_top_1])) + "\n")
            cumulative_corrections += int((1 - update_top_1) * 50)
            pivot_file.write(",".join(map(str, [model, dataset, split, iteration, "update", "cumulative_corrections", cumulative_corrections])) + "\n")

        iteration += 1


with open("pivot.txt", "w") as pivot_file:
    pivot_file.write("model,dataset,split,iteration,predicted_on,type,value\n")
    for dataset in datasets:
        for split in splits:
            for model in od_models:
                analyse_od(model, dataset, split, pivot_file)
            for model in ic_models:
                analyse_ic(model, dataset, split, pivot_file)
            for model in hybrid_models:
                analyse_hybrid(model, dataset, split, pivot_file)
            for model in active_models:
                analyse_active(model, dataset, split, pivot_file)
