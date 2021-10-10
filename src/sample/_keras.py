from collections import OrderedDict

from pandas import DataFrame
from tensorflow import keras

from ._types import Dataset, Predictions, LabelIndices
from ._util import label_indices


def data_flow_from_disk(
        path: str,
        dataset: Dataset,
        label_indices: LabelIndices,
        shuffle: bool,
        batch_size: int,
        seed: int,
        model: str
):
    if model == "resnet50" or model == "resnet152":
        preprocessing_function = keras.applications.resnet.preprocess_input
    elif model == "mobilenet":
        preprocessing_function = keras.applications.mobilenet.preprocess_input
    else:
        raise Exception(f"Unknown model {model}")

    gen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )

    dataframe = DataFrame(
        data={
            "filename": list(dataset.keys()),
            "class": list(label_indices[label] for label in dataset.values())
        },
        columns=["filename", "class"]
    )

    return gen.flow_from_dataframe(
        dataframe,
        path,
        target_size=(224, 224),
        class_mode="raw",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed
    )


def model_for_fine_tuning(model: str, num_labels: int, weights: str) -> keras.models.Model:
    if model == "resnet50":
        return ResNet50_for_fine_tuning(num_labels, weights)
    elif model == "resnet152":
        return ResNet152_for_fine_tuning(num_labels, weights)
    elif model == "mobilenet":
        return MobileNet_for_fine_tuning(num_labels, weights)
    else:
        raise Exception(f"Unknown model {model}")


def ResNet50_for_fine_tuning(num_labels: int, weights: str) -> keras.models.Model:
    base_model = keras.applications.ResNet50(include_top=False, weights=weights)
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    fine_tuning_model = base_model(inputs, training=False)
    fine_tuning_model = keras.layers.AveragePooling2D(pool_size=(7, 7))(fine_tuning_model)
    fine_tuning_model = keras.layers.Flatten(name="flatten")(fine_tuning_model)
    fine_tuning_model = keras.layers.Dense(256, activation="relu")(fine_tuning_model)
    fine_tuning_model = keras.layers.Dropout(0.5)(fine_tuning_model)
    fine_tuning_model = keras.layers.Dense(num_labels, activation="softmax")(fine_tuning_model)

    return keras.models.Model(inputs=inputs, outputs=fine_tuning_model)


def ResNet152_for_fine_tuning(num_labels: int, weights: str) -> keras.models.Model:
    base_model = keras.applications.ResNet152(include_top=False, weights=weights)
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    fine_tuning_model = base_model(inputs, training=False)
    fine_tuning_model = keras.layers.AveragePooling2D(pool_size=(7, 7))(fine_tuning_model)
    fine_tuning_model = keras.layers.Flatten(name="flatten")(fine_tuning_model)
    fine_tuning_model = keras.layers.Dense(256, activation="relu")(fine_tuning_model)
    fine_tuning_model = keras.layers.Dropout(0.5)(fine_tuning_model)
    fine_tuning_model = keras.layers.Dense(num_labels, activation="softmax")(fine_tuning_model)

    return keras.models.Model(inputs=inputs, outputs=fine_tuning_model)


def MobileNet_for_fine_tuning(num_labels: int, weights: str) -> keras.models.Model:
    base_model = keras.applications.MobileNet(include_top=False, weights=weights)
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    fine_tuning_model = base_model(inputs, training=False)
    fine_tuning_model = keras.layers.AveragePooling2D(pool_size=(7, 7))(fine_tuning_model)
    fine_tuning_model = keras.layers.Flatten(name="flatten")(fine_tuning_model)
    fine_tuning_model = keras.layers.Dense(256, activation="relu")(fine_tuning_model)
    fine_tuning_model = keras.layers.Dropout(0.5)(fine_tuning_model)
    fine_tuning_model = keras.layers.Dense(num_labels, activation="softmax")(fine_tuning_model)

    return keras.models.Model(inputs=inputs, outputs=fine_tuning_model)


class MyLogger(keras.callbacks.Callback):
    def __init__(self, batch_size: int, num_items: int):
        self._batch_size = batch_size
        self._num_items = num_items

    def on_predict_batch_end(self, batch, logs=None):
        print(f"predicted {(batch + 1) * self._batch_size} of {self._num_items}")


def dataset_predictions(model_name: str, path: str, dataset: Dataset) -> Predictions:
    if model_name == "resnet50":
        model = keras.applications.ResNet50()
    elif model_name == "resnet152":
        model = keras.applications.ResNet152()
    elif model_name == "mobilenet":
        model = keras.applications.MobileNet()
    else:
        raise Exception(f"Unknown model {model_name}")

    model = keras.models.Model(model.input, model.layers[-2].output)
    dataset_size = len(dataset)

    predictions = model.predict(
        data_flow_from_disk(
            path,
            dataset,
            label_indices(dataset),  # Doesn't actually matter
            False,
            5,
            0,
            model_name
        ),
        callbacks=[MyLogger(5, dataset_size)]
    )

    result = OrderedDict()
    for i, filename in enumerate(dataset.keys()):
        result[filename] = predictions[i]

    return result
