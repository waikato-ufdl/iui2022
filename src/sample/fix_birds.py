import os

from wai.annotations.core.builder import ConversionPipelineBuilder
from wai.annotations.domain.image import Image
from wai.annotations.domain.image.object_detection import ImageObjectDetectionInstance
from wai.common.adams.imaging.locateobjects import LocatedObject, LocatedObjects
from wai.annotations.domain.image.object_detection.util import set_object_label

images = {}
with open("images.txt", "r") as file:
    for line in file.readlines():
        id, filename = line.strip().split(" ")
        images[id] = filename

bounding_boxes = {}
with open("bounding_boxes.txt", "r") as file:
    for line in file.readlines():
        id, x, y, width, height = line.strip().split(" ")
        bounding_boxes[id] = LocatedObject(int(float(x)), int(float(y)), int(float(width)), int(float(height)))

classes = {}
with open("classes.txt", "r") as file:
    for line in file.readlines():
        id, class_ = line.strip().split(" ")
        classes[id] = class_

image_class_labels = {}
with open("image_class_labels.txt", "r") as file:
    for line in file.readlines():
        id, class_id = line.strip().split(" ")
        image_class_labels[id] = class_id

dataset = {}
i = 0
for image_id, filename in images.items():
    class_id = image_class_labels[image_id]
    class_ = classes[class_id]
    file_data = Image.from_file(os.path.join("subdir", filename))
    bbox = bounding_boxes[image_id]
    set_object_label(bbox, class_)
    dataset[filename] = ImageObjectDetectionInstance(file_data, LocatedObjects((bbox,)))
    i += 1
    print(f"{i} {filename}")

pipeline = ConversionPipelineBuilder.from_options([
    "to-voc-od",
    "-o",
    "voc"
])

pipeline.process(source=dataset.values())
