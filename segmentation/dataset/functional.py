import numpy as np
import cv2
MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

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

def transpose(img):
    return img.transpose(1, 0, 2) if len(img.shape)>2 else img.transpose(1, 0)

def brightness_and_constrast(img, alpha = 1, beta = 0):
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    if dtype == np.uint8:
        lut = np.arange(0, max_value+1).astype("float32")
        if alpha != 1:
            lut*= alpha
        if beta != 0 :
            lut += beta * np.mean(img)
        lut = np.clip(lut, 0, max_value).astype(dtype)
        img = cv2.LUT(img, lut)
    else:
        if alpha != 1:
            img *= alpha
        if beta!=0:
            img += beta*np.mean(img)
    return img

def clahe(img, cliplimit = 3, tileGridSize = (8, 8)):
    clahe = cv2.createCLAHE(cliplimit, tileGridSize)
    return clahe.apply(img)