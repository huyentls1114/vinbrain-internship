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
