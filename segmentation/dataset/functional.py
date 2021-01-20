import numpy as np


def crop(img, crop_height, crop_width, h_start, w_start):
    h, w = img.shape[:2]
    if h < crop_height or w < crop_width:
        raise ValueError("Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=h, width=w
            ))
    y1 = int((h-crop_height)*h_start)
    y2 = y1+crop_height
    x1 = int((w - crop_width)*w_start)
    x2 = x1 + crop_width
    return img[y1:y2, x1:x2]
    
def vertical_flip(img):
    return np.ascontiguousarray(img[::-1, ...])
def horizontal_flip(img):
    return np.ascontiguousarray(img[:,::-1, ...])