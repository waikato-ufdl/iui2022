import os
import shutil
import sys
from random import Random

from wai.annotations.core.util import chain_map
from wai.annotations.main import main as wai_annotations_main

from sample import *
from sample.splitters import RandomSplitter, TopNSplitter

INIT_LR = 1e-4
BS = 5
SEED = 42
VALIDATION_PERCENT = 0.15
LR = 0.02

RANDOM = Random(SEED)
SHUFFLE_RANDOM = Random(SEED)

SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])
NUM_EPOCHS = int(sys.argv[2])
MODEL = sys.argv[3]
RELATIVE_DIR = os.path.join(SOURCE_PATH, sys.argv[4])
GPU = sys.argv[5]
IC_MODEL = sys.argv[6]

CWD = os.getcwd()

schedule_dataset = load_dataset("schedule.txt")

source_dataset = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))
label_indices = label_indices(source_dataset)

PREDICTIONS_FILE_HEADER = predictions_file_header(label_indices)

MODEL_DIR = os.path.join(CWD, f"{MODEL}-{IC_MODEL}-predicted")

os.makedirs(MODEL_DIR)

with open(f"{MODEL_DIR}/setup.py", "w") as file:
    file.write(f"NUM_EPOCHS = {NUM_EPOCHS}\n")
    file.write(f"LR = {LR}\n")
    with open(os.path.join(CWD, '..', f'setup_{MODEL}_{SOURCE_DATASET}.py')) as source_file:
        file.writelines(source_file.readlines())

with open(f"{MODEL_DIR}/labels.txt", "w") as file:
    file.write(",".join(label_indices.keys()))

with open(f"{MODEL_DIR}/object_labels.txt", "w") as file:
    file.write("object")

holdout_dataset = load_dataset("holdout.txt")

splitter = TopNSplitter(50)

iteration = 0
iteration_dataset, remaining_dataset = splitter(schedule_dataset)
while True:
    # Debug
    print(f"ITERATION {iteration}")

    # Create a new directory for this iteration
    ITERATION_DIR = os.path.join(MODEL_DIR, str(iteration))
    os.makedirs(ITERATION_DIR)

    # Select a random subset for validation
    validation_size = max(int(len(iteration_dataset) * VALIDATION_PERCENT), 1)
    validation_dataset, train_dataset = RandomSplitter(validation_size, RANDOM)(iteration_dataset)

    # Randomly re-order the training dataset
    train_dataset = shuffle_dataset(train_dataset, SHUFFLE_RANDOM)

    write_dataset(change_path(validation_dataset, RELATIVE_DIR), f"{ITERATION_DIR}/validation.txt")
    write_dataset(change_path(train_dataset, RELATIVE_DIR), f"{ITERATION_DIR}/train.txt")

    # Train the boxer on the data, with only 'object' labels, in the first iteration
    if iteration == 0:
        os.makedirs(f"{ITERATION_DIR}/val")
        os.makedirs(f"{ITERATION_DIR}/train")
        os.makedirs(f"{ITERATION_DIR}/output")
        wai_annotations_main([
            "convert",
            "from-voc-od",
            "-I",
            f"{ITERATION_DIR}/validation.txt",
            "map-labels",
            *chain_map(lambda label: ("-m", f"{label}=object"), label_indices.keys()),
            "to-coco-od",
            "-o",
            f"{ITERATION_DIR}/val/annotations.json",
            "--pretty",
            "--categories",
            "object"
        ])
        wai_annotations_main([
            "convert",
            "from-voc-od",
            "-I",
            f"{ITERATION_DIR}/train.txt",
            "map-labels",
            *chain_map(lambda label: ("-m", f"{label}=object"), label_indices.keys()),
            "to-coco-od",
            "-o",
            f"{ITERATION_DIR}/train/annotations.json",
            "--pretty",
            "--categories",
            "object"
        ])
        run_command(
            f"docker run "
            f"--gpus device={GPU} "
            f"--shm-size 8G "
            f"-u $(id -u):$(id -g) "
            f"-e USER=$USER "
            f"-e MMDET_CLASSES=\"'/labels.txt'\" "
            f"-e MMDET_OUTPUT=/data/output "
            f"-v {MODEL_DIR}/object_labels.txt:/labels.txt "
            f"-v {MODEL_DIR}/setup.py:/setup.py "
            f"-v {os.path.join(CWD, '..', f'base_{MODEL}.pth')}:/model.pth "
            f"-v {ITERATION_DIR}:/data "
            f"public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_cuda10 "
            f"mmdet_train /setup.py"
        )

    # Use boxer to predict bboxes for the training dataset
    os.makedirs(f"{ITERATION_DIR}/training_predictions")
    os.makedirs(f"{ITERATION_DIR}/training_predictions_in")
    wai_annotations_main([
        "convert",
        "from-voc-od",
        "-I",
        f"{ITERATION_DIR}/train.txt",
        "map-labels",
        *chain_map(lambda label: ("-m", f"{label}=object"), label_indices.keys()),
        "to-coco-od",
        "-o",
        f"{ITERATION_DIR}/training_predictions_in/annotations.json",
        "--pretty",
        "--categories",
        "object"
    ])
    run_command(
        f"docker run "
        f"--gpus device={GPU} "
        f"--shm-size 8G "
        f"-u $(id -u):$(id -g) "
        f"-e USER=$USER "
        f"-e MMDET_CLASSES=\"'/labels.txt'\" "
        f"-e MMDET_OUTPUT=/data_in/output "
        f"-v {MODEL_DIR}/object_labels.txt:/labels.txt "
        f"-v {MODEL_DIR}/setup.py:/setup.py "
        f"-v {os.path.join(CWD, '..', f'base_{MODEL}.pth')}:/model.pth "
        f"-v {ITERATION_DIR}:/data "
        f"-v {os.path.join(MODEL_DIR, '0')}:/data_in "
        f"public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_cuda10 "
        f"mmdet_predict "
        f"--checkpoint /data_in/output/latest.pth "
        f"--config /setup.py "
        f"--prediction_in /data/training_predictions_in/ "
        f"--prediction_out /data/training_predictions/ "
        f"--labels /labels.txt "
        f"--score 0 "
        f"--delete_input"
    )

    # Use boxer to predict bboxes for the validation dataset
    os.makedirs(f"{ITERATION_DIR}/val_predictions")
    os.makedirs(f"{ITERATION_DIR}/val_predictions_in")
    wai_annotations_main([
        "convert",
        "from-voc-od",
        "-I",
        f"{ITERATION_DIR}/validation.txt",
        "map-labels",
        *chain_map(lambda label: ("-m", f"{label}=object"), label_indices.keys()),
        "to-coco-od",
        "-o",
        f"{ITERATION_DIR}/val_predictions_in/annotations.json",
        "--pretty",
        "--categories",
        "object"
    ])
    run_command(
        f"docker run "
        f"--gpus device={GPU} "
        f"--shm-size 8G "
        f"-u $(id -u):$(id -g) "
        f"-e USER=$USER "
        f"-e MMDET_CLASSES=\"'/labels.txt'\" "
        f"-e MMDET_OUTPUT=/data_in/output "
        f"-v {MODEL_DIR}/object_labels.txt:/labels.txt "
        f"-v {MODEL_DIR}/setup.py:/setup.py "
        f"-v {os.path.join(CWD, '..', f'base_{MODEL}.pth')}:/model.pth "
        f"-v {ITERATION_DIR}:/data "
        f"-v {os.path.join(MODEL_DIR, '0')}:/data_in "
        f"public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_cuda10 "
        f"mmdet_predict "
        f"--checkpoint /data_in/output/latest.pth "
        f"--config /setup.py "
        f"--prediction_in /data/val_predictions_in/ "
        f"--prediction_out /data/val_predictions/ "
        f"--labels /labels.txt "
        f"--score 0 "
        f"--delete_input"
    )

    # Use boxer to predict bboxes for the holdout dataset
    write_dataset(change_path(holdout_dataset, RELATIVE_DIR), f"{ITERATION_DIR}/holdout.txt")
    os.makedirs(f"{ITERATION_DIR}/predictions")
    os.makedirs(f"{ITERATION_DIR}/predictions_in")
    wai_annotations_main([
        "convert",
        "from-voc-od",
        "-I",
        f"{ITERATION_DIR}/holdout.txt",
        "map-labels",
        *chain_map(lambda label: ("-m", f"{label}=object"), label_indices.keys()),
        "to-coco-od",
        "-o",
        f"{ITERATION_DIR}/predictions_in/annotations.json",
        "--pretty",
        "--categories",
        "object"
    ])
    run_command(
        f"docker run "
        f"--gpus device={GPU} "
        f"--shm-size 8G "
        f"-u $(id -u):$(id -g) "
        f"-e USER=$USER "
        f"-e MMDET_CLASSES=\"'/labels.txt'\" "
        f"-e MMDET_OUTPUT=/data_in/output "
        f"-v {MODEL_DIR}/object_labels.txt:/labels.txt "
        f"-v {MODEL_DIR}/setup.py:/setup.py "
        f"-v {os.path.join(CWD, '..', f'base_{MODEL}.pth')}:/model.pth "
        f"-v {ITERATION_DIR}:/data "
        f"-v {os.path.join(MODEL_DIR, '0')}:/data_in "
        f"public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_cuda10 "
        f"mmdet_predict "
        f"--checkpoint /data_in/output/latest.pth "
        f"--config /setup.py "
        f"--prediction_in /data/predictions_in/ "
        f"--prediction_out /data/predictions/ "
        f"--labels /labels.txt "
        f"--score 0 "
        f"--delete_input"
    )

    # If there is going to be another iteration, also predict bboxes for update dataset
    update_dataset = None
    if len(remaining_dataset) != 0:
        update_dataset, remaining_dataset = splitter(remaining_dataset)
        write_dataset(change_path(update_dataset, RELATIVE_DIR), f"{ITERATION_DIR}/update.txt")
        os.makedirs(f"{ITERATION_DIR}/update_predictions")
        os.makedirs(f"{ITERATION_DIR}/update_predictions_in")
        wai_annotations_main([
            "convert",
            "from-voc-od",
            "-I",
            f"{ITERATION_DIR}/update.txt",
            "map-labels",
            *chain_map(lambda label: ("-m", f"{label}=object"), label_indices.keys()),
            "to-coco-od",
            "-o",
            f"{ITERATION_DIR}/update_predictions_in/annotations.json",
            "--pretty",
            "--categories",
            "object"
        ])
        run_command(
            f"docker run "
            f"--gpus device={GPU} "
            f"--shm-size 8G "
            f"-u $(id -u):$(id -g) "
            f"-e USER=$USER "
            f"-e MMDET_CLASSES=\"'/labels.txt'\" "
            f"-e MMDET_OUTPUT=/data_in/output "
            f"-v {MODEL_DIR}/object_labels.txt:/labels.txt "
            f"-v {MODEL_DIR}/setup.py:/setup.py "
            f"-v {os.path.join(CWD, '..', f'base_{MODEL}.pth')}:/model.pth "
            f"-v {ITERATION_DIR}:/data "
            f"-v {os.path.join(MODEL_DIR, '0')}:/data_in "
            f"public.aml-repo.cms.waikato.ac.nz:443/open-mmlab/mmdetection:2020-03-01_cuda10 "
            f"mmdet_predict "
            f"--checkpoint /data_in/output/latest.pth "
            f"--config /setup.py "
            f"--prediction_in /data/update_predictions_in/ "
            f"--prediction_out /data/update_predictions/ "
            f"--labels /labels.txt "
            f"--score 0 "
            f"--delete_input"
        )

    # Write the cropped training/validation datasets
    bboxes = get_highest_score_bbox(
        f"{ITERATION_DIR}/training_predictions",
        train_dataset
    )
    for filename, label in train_dataset.items():
        dest_path = f"{ITERATION_DIR}/train_cropped/{label}"
        os.makedirs(dest_path, exist_ok=True)
        if filename in bboxes:
            crop_image(
                os.path.join(SOURCE_PATH, filename),
                dest_path,
                bboxes[filename]
            )
        else:
            shutil.copy(
                os.path.join(SOURCE_PATH, filename),
                dest_path
            )
    write_dataset(change_path(train_dataset, "train_cropped", True, None), f"{ITERATION_DIR}/train_cropped.txt")
    bboxes = get_highest_score_bbox(
        f"{ITERATION_DIR}/val_predictions",
        validation_dataset
    )
    for filename, label in validation_dataset.items():
        dest_path = f"{ITERATION_DIR}/validation_cropped/{label}"
        os.makedirs(dest_path, exist_ok=True)
        if filename in bboxes:
            crop_image(
                os.path.join(SOURCE_PATH, filename),
                dest_path,
                bboxes[filename]
            )
        else:
            shutil.copy(
                os.path.join(SOURCE_PATH, filename),
                dest_path
            )
    write_dataset(change_path(validation_dataset, "validation_cropped", True, None), f"{ITERATION_DIR}/validation_cropped.txt")

    # Crop the holdout dataset using the bboxes predicted by the boxer
    bboxes = get_highest_score_bbox(
        f"{ITERATION_DIR}/predictions",
        holdout_dataset
    )
    for filename, label in holdout_dataset.items():
        dest_path = f"{ITERATION_DIR}/holdout_cropped/{label}"
        os.makedirs(dest_path, exist_ok=True)
        if filename in bboxes:
            crop_image(
                os.path.join(SOURCE_PATH, filename),
                dest_path,
                bboxes[filename]
            )
        else:
            shutil.copy(
                os.path.join(SOURCE_PATH, filename),
                dest_path
            )
    write_dataset(change_path(holdout_dataset, "holdout_cropped", True, None), f"{ITERATION_DIR}/holdout_cropped.txt")

    # If there is going to be another iteration, also crop the update dataset
    if update_dataset is not None:
        bboxes = get_highest_score_bbox(
            f"{ITERATION_DIR}/update_predictions",
            update_dataset
        )
        for filename, label in update_dataset.items():
            dest_path = f"{ITERATION_DIR}/update_cropped/{label}"
            os.makedirs(dest_path, exist_ok=True)
            if filename in bboxes:
                crop_image(
                    os.path.join(SOURCE_PATH, filename),
                    dest_path,
                    bboxes[filename]
                )
            else:
                shutil.copy(
                    os.path.join(SOURCE_PATH, filename),
                    dest_path
                )
        write_dataset(change_path(update_dataset, f"update_cropped", True, None), f"{ITERATION_DIR}/update_cropped.txt")

    # Use the associated Docker image to perform classification steps
    predict_datasets = "/holdout_cropped.txt" if update_dataset is None else "/holdout_cropped.txt /update_cropped.txt"
    predict_volumes = (
        f"-v {ITERATION_DIR}/holdout_cropped:/holdout_cropped "
        f"-v {ITERATION_DIR}/holdout_cropped.txt:/holdout_cropped.txt"
        if update_dataset is None else
        f"-v {ITERATION_DIR}/holdout_cropped:/holdout_cropped "
        f"-v {ITERATION_DIR}/holdout_cropped.txt:/holdout_cropped.txt "
        f"-v {ITERATION_DIR}/update_cropped:/update_cropped "
        f"-v {ITERATION_DIR}/update_cropped.txt:/update_cropped.txt"
    )
    run_command(
        f"docker run "
        f"--gpus device={GPU} "
        f"--shm-size 8G "
        f"-u $(id -u):$(id -g) "
        f"-e USER=$USER "
        f"{predict_volumes} "
        f"-v {ITERATION_DIR}/train_cropped:/train_cropped "
        f"-v {ITERATION_DIR}/train_cropped.txt:/train.txt "
        f"-v {ITERATION_DIR}/validation_cropped:/validation_cropped "
        f"-v {ITERATION_DIR}/validation_cropped.txt:/val.txt "
        f"-v {sys.argv[1]}:/dataset.txt "
        f"-v {ITERATION_DIR}:/output "
        f"fine_tune_hybrid_ic "
        f"{NUM_EPOCHS} "
        f"{IC_MODEL} "
        f"{predict_datasets}"
    )

    # Clean up
    rm_dir(f"{ITERATION_DIR}/predictions_in")
    rm_dir(f"{ITERATION_DIR}/update_predictions_in")
    rm_dir(f"{ITERATION_DIR}/training_predictions_in")
    rm_dir(f"{ITERATION_DIR}/val_predictions_in")
    os.remove(f"{ITERATION_DIR}/validation.txt")
    os.remove(f"{ITERATION_DIR}/train.txt")
    os.remove(f"{ITERATION_DIR}/update.txt")
    os.remove(f"{ITERATION_DIR}/holdout.txt")

    iteration_dataset = merge(iteration_dataset, update_dataset)

    iteration += 1
