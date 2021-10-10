import os
import sys
from collections import OrderedDict
from typing import List
from xml.etree.ElementTree import ElementTree as XMLTree

from defusedxml import ElementTree

from wai.annotations.voc.od.component import FromVOCOD
from wai.common.adams.imaging.locateobjects import LocatedObject

from sample import *


SOURCE_PATH, SOURCE_DATASET, SOURCE_EXT = split_arg(sys.argv[1])

RELATIVE_DIR = os.path.join(SOURCE_PATH, sys.argv[2])

DEST = os.path.join(SOURCE_PATH, f"{SOURCE_DATASET}_top_half{SOURCE_EXT}")

source_dataset = load_dataset(os.path.join(SOURCE_PATH, SOURCE_DATASET + SOURCE_EXT))

relative_source_dataset = change_path(source_dataset, RELATIVE_DIR)

by_label = per_label(source_dataset)

with_areas = OrderedDict((label, []) for label in by_label)


def get_area(filename: str) -> float:
    changed_filename = change_filename(filename, RELATIVE_DIR)

    voc_file: XMLTree = ElementTree.parse(changed_filename)

    located_objects: List[LocatedObject] = list(map(FromVOCOD.to_located_object, voc_file.findall("object")))

    if len(located_objects) != 1:
        raise Exception(f"{len(located_objects)} objects in {filename}")

    area = located_objects[0].width * located_objects[0].height

    if area <= 0.0:
        raise Exception(f"{filename} has area {area}")

    return area


for label, entries in by_label.items():
    for filename in entries:
        with_areas[label].append((filename, get_area(filename)))

    with_areas[label].sort(key=lambda x: x[1], reverse=True)

    with_areas[label] = set(filename for filename, area in with_areas[label][:(len(with_areas[label]) + 1) // 2])

top_half_dataset = OrderedDict()

for filename, label in source_dataset.items():
    if filename in with_areas[label]:
        top_half_dataset[filename] = label

write_dataset(top_half_dataset, DEST)