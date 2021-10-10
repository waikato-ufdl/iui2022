import itertools
import sys
from collections import OrderedDict

from matplotlib import pyplot

TEXT_REPLACEMENTS = {
    "asl": "ASL",
    "flowers-files": "Flowers",
    "dog_breeds": "Dogs",
    "birds": "Birds",
    "holdout_accuracy": "Holdout Accuracy",
    "update_cumulative_corrections": "Update Cumulative Corrections",
    "mobilenet": "Mobilenet",
    "resnet50": "Resnet50",
    "resnet152": "Resnet152",
    "accuracy": "Accuracy",
    "rand": "Random ordering",
    "active-ranked": "Uncertainty sampling",
    "active-softmaxBALD-0.125": "Stochastic acquisition",
    "active-kh": "Kernel herding (active)",
    "kh": "Kernel herding (a priori)",
    "cumulative_corrections": "Cumulative Corrections"
}

MIN_EXTENTS = True


def capitalise_all(string: str) -> str:
    for orig, repl in TEXT_REPLACEMENTS.items():
        string = string.replace(orig, repl)
    return string


LABEL_SORT_ORDER=[
    "Random ordering",
    "Kernel herding (a priori)",
    "Uncertainty sampling",
    "Stochastic acquisition",
    "Kernel herding (active)"
]


def sort_labels(labels):
    sorted = list(zip(*labels))
    sorted.sort(key=lambda line_and_label: LABEL_SORT_ORDER.index(line_and_label[1]))
    result = ([], [])
    for line, label in sorted:
        result[0].append(line)
        result[1].append(label)
    return result


SOURCE = sys.argv[1]

results = OrderedDict()

with open(SOURCE, "r") as source_file:
    source_file.readline()
    for line in source_file.readlines():
        model, dataset, split, iteration, predicted_on, type, value = line.strip().split(",")
        prediction = f"{predicted_on}_{type}"
        iteration = int(iteration)
        value = float(value)
        if "." in model:
            if split != "uni":
                continue
            model, split = model.split(".", 1)
            split = f"active-{split}"
        elif split == "uni":
            continue

        if dataset not in results:
            dataset_dict = OrderedDict()
            results[dataset] = dataset_dict
        else:
            dataset_dict = results[dataset]

        if model not in dataset_dict:
            model_dict = OrderedDict()
            dataset_dict[model] = model_dict
        else:
            model_dict = dataset_dict[model]

        if split not in model_dict:
            split_dict = OrderedDict()
            model_dict[split] = split_dict
        else:
            split_dict = model_dict[split]

        if prediction not in split_dict:
            prediction_list = []
            split_dict[prediction] = prediction_list
        else:
            prediction_list = split_dict[prediction]

        while len(prediction_list) <= iteration:
            prediction_list.append(0.0)

        prediction_list[iteration] = value

if MIN_EXTENTS:
    for dataset, dataset_dict in results.items():
        min_extent = sys.maxsize
        for model, model_dict in dataset_dict.items():
            for split, split_dict in model_dict.items():
                for prediction, prediction_list in split_dict.items():
                    min_extent = min(min_extent, len(prediction_list))
        for model, model_dict in dataset_dict.items():
            for split, split_dict in model_dict.items():
                for prediction, prediction_list in split_dict.items():
                    while len(prediction_list) > min_extent:
                        prediction_list.pop()


for TARGET in ["holdout_accuracy", "update_cumulative_corrections"]:
    Y_LABEL = TARGET[TARGET.index("_") + 1:]

    fig, axes = pyplot.subplots(
        4, 3,
        sharey="row", sharex="row",
        subplot_kw={"xmargin": 0, "ymargin": 0},
        gridspec_kw={"bottom": 0.12, "top": 0.92, "left": 0.07, "right": 0.93, "hspace": 0.65, "wspace": 0.1}
    )
    fig.suptitle(capitalise_all(TARGET))
    fig.supylabel("Dataset")
    fig.supxlabel("Model", y=0.05)
    colour_cycle: itertools.cycle = pyplot.rcParams['axes.prop_cycle']()
    split_colours = {}
    axes_iter = axes.flat
    lines = []
    for dataset, dataset_dict in results.items():
        for model, model_dict in dataset_dict.items():
            axis = next(axes_iter)
            axis.set_title(capitalise_all(f"{model} - {dataset}"))
            axis.set_xlabel('Iteration')
            axis.set_ylabel(capitalise_all(Y_LABEL))
            for split, split_dict in model_dict.items():
                if split not in split_colours:
                    split_colours[split] = next(colour_cycle)['color']
                prediction_list = split_dict[TARGET]
                lines.append(axis.plot(prediction_list, label=capitalise_all(split), color=split_colours[split]))
            axis.set_ybound(lower=0.0, upper=1.0 if Y_LABEL == "accuracy" else None)

    fig.legend(
        *sort_labels(axis.get_legend_handles_labels()),
        loc="lower center",
        ncol=5,
        fontsize="large"
    )
    pyplot.show()

##
# Accuracy vs. Corrections
##
fig, axes = pyplot.subplots(
    4, 3,
    sharey="row", sharex="row",
    subplot_kw={"xmargin": 0, "ymargin": 0},
    gridspec_kw={"bottom": 0.12, "top": 0.92, "left": 0.07, "right": 0.93, "hspace": 0.65, "wspace": 0.1}
)
fig.suptitle("Accuracy vs. Corrections")
fig.supylabel("Dataset")
fig.supxlabel("Model", y=0.05)
colour_cycle: itertools.cycle = pyplot.rcParams['axes.prop_cycle']()
split_colours = {}
axes_iter = axes.flat
lines = []
for dataset, dataset_dict in results.items():
    for model, model_dict in dataset_dict.items():
        axis = next(axes_iter)
        axis.set_title(capitalise_all(f"{model} - {dataset}"))
        axis.set_xlabel("Corrections")
        axis.set_ylabel("Accuracy")
        for split, split_dict in model_dict.items():
            if split not in split_colours:
                split_colours[split] = next(colour_cycle)['color']
            accuracy_list = split_dict["holdout_accuracy"]
            corrections_list = split_dict["update_cumulative_corrections"]
            if len(accuracy_list) == len(corrections_list) + 1:
                accuracy_list = accuracy_list[:-1]
            lines.append(axis.plot(corrections_list, accuracy_list, label=capitalise_all(split), color=split_colours[split]))
        axis.set_ybound(lower=0.0, upper=1.0)

fig.legend(
    *sort_labels(axis.get_legend_handles_labels()),
    loc="lower center",
    ncol=5,
    fontsize="large"
)
pyplot.show()

##
# Non-cumulative corrections vs. iteration
##
fig, axes = pyplot.subplots(
    4, 3,
    sharey="row", sharex="row",
    subplot_kw={"xmargin": 0, "ymargin": 0},
    gridspec_kw={"bottom": 0.12, "top": 0.92, "left": 0.07, "right": 0.93, "hspace": 0.65, "wspace": 0.1}
)
fig.suptitle("Corrections")
fig.supylabel("Dataset")
fig.supxlabel("Model", y=0.05)
colour_cycle: itertools.cycle = pyplot.rcParams['axes.prop_cycle']()
split_colours = {}
axes_iter = axes.flat
lines = []
for dataset, dataset_dict in results.items():
    for model, model_dict in dataset_dict.items():
        axis = next(axes_iter)
        axis.set_title(capitalise_all(f"{model} - {dataset}"))
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Corrections")
        for split, split_dict in model_dict.items():
            if split not in split_colours:
                split_colours[split] = next(colour_cycle)['color']
            corrections_list = split_dict["update_cumulative_corrections"]
            corrections_list = [
                corrections_list[i] - corrections_list[i-1] if i > 0 else corrections_list[i]
                for i in range(len(corrections_list))
            ]
            lines.append(axis.plot(corrections_list, label=capitalise_all(split), color=split_colours[split]))
        axis.set_ybound(lower=0.0, upper=50.0)

fig.legend(
    *sort_labels(axis.get_legend_handles_labels()),
    loc="lower center",
    ncol=5,
    fontsize="large"
)
pyplot.show()