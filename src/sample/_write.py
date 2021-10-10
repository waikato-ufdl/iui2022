from ._types import Dataset, Predictions


def write_dataset(dataset: Dataset, filename: str):
    """
    TODO
    """
    with open(filename, "w") as file:
        for f in dataset.keys():
            file.write(f + "\n")


def write_predictions(predictions: Predictions, header: str, filename: str):
    """
    TODO
    """
    with open(filename, "w") as file:
        file.write(header)
        for item, prediction in predictions.items():
            file.write(
                f"{item},{','.join(map(str, prediction))}\n"
            )
