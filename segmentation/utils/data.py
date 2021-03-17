import numpy as np
def process_img(image, channel = 3):
    "return image with shape [w, h, channel]"
    # print(image.shape)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        return np.dstack((image, )*channel)
    if (channel >1) and (image.shape[2] == 1):
        return np.dstack((image, )*channel)
    if (channel == 1) and (image.shape[2] == 3):
        return image[:,:, 0:1]
    if image.shape[2] == channel:
        return image
    if image.shape[2] > channel:
        return image[:,:,:channel]

import cv2
def resize(image, w = None, h = None, scale = None):
    h_origin, w_origin = image.shape[:2]
    print(h_origin, w_origin)
    if w is not None:
        if h is None:
            h = int(w/w_origin*h_origin)
            return cv2.resize(image, (w, h))
        else:
            return cv2.resize(image, w, h)
    if h is not None:
        if w is None:
            w = int(h/h_origin*w_origin)
            return cv2.resize(image, (w, h))
    if scale is not None:
        w = int(w_origin*scale)
        h = int(h_origin*scale)
        return cv2.resize(image, (w, h))
    return image

def center_crop(image):
    #assum w>>h
    h_origin, w_origin = image.shape[:2]

    if w_origin > h_origin:
        w_1 = int(w_origin//2 - h_origin//2)
        w_2 = int(w_origin//2 + h_origin//2)
        return image[:, w_1: w_2] 
    elif w_origin < h_origin:
        h_1 = int(h_origin//2 - w_origin//2)
        h_2 = int(h_origin//2 + w_origin//2)
        return image[h_1: h_2, :] 
    else:
        return image

    