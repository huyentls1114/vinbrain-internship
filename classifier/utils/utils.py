import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook= OrderedDict)


def show_img(image):
    "show an image tensor"
    image = (image/2)+0.5
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()
