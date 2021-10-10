import os
from typing import Tuple

from PIL import Image


def crop_image(
        filename: str,
        dest_path: str,
        region: Tuple[int, int, int, int]
):
    """
    TODO
    """
    source_path, source_filename = os.path.split(filename)
    dest = os.path.join(dest_path, source_filename)
    image = Image.open(filename)
    image = image.crop(region)
    image.save(dest)
