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

def save_loss_to_file(file_, epoch, step, loss_train, loss_val, acc_val, lr):
    file_ = open(file_, "a+")
    file_.writelines("Epoch %d step %d\n"%(epoch, step))
    file_.writelines("\tLoss average %f\n"%(loss_train))
    file_.writelines("\tLoss valid average %f, acc valid %f\n"%(loss_val, acc_val))
    file_.writelines("learning_rate %f\n"%(lr))

def steps_per_epoch_train(dataset_dict, split_train_val, batch_size):
    DatasetClass = dataset_dict["class"]
    train_dataset = DatasetClass(dataset_dict["argument"],transform = transform, mode = "train")
    return len(train_dataset)*split_train_val/batch_size