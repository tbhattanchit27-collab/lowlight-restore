import cv2
import numpy as np


def read_rgb(path, max_side=None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def save_rgb(path, img):
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
