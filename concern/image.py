import numpy as np
import cv2
from smart_path import smart_path


def load_image(path: smart_path):
    with path.open("rb") as reader:
        data = np.fromstring(reader.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return
        img = img[..., ::-1]
    return img


