import numpy as np
import cv2

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

def rotate(img, angle, interpolation = cv2.INTER_LINEAR, boder_mode= cv2.BORDER_REFLECT_101, value = None):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, scale = 1.0)
    return cv2.warpAffine(img, M = matrix, dsize=(w, h), flags=interpolation, borderMode=boder_mode, borderValue=value)

