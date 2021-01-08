import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def show_img(image):
    "show an image tensor"
    image = (image/2)+0.5
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()

def conver_numpy_image(image):
    image = (image/2)+0.5
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    return image
def contour(images, masks):
    main = images.copy()
    _,contours,_ = cv2.findContours(masks,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i,c in enumerate(contours):
        colour = RGBforLabel.get(2)
        cv2.drawContours(main,[c],-1,colour,1)
    return main

def save_loss_to_file(file_, epoch, step, loss_train, loss_val, acc_val, lr):
    '''
    target: save loss to the file
    input:
        - file_: file contain loss
        - epoch: Interger
        - Step: Interger
        - loss_train, loss_val, acc_val: float
        - lr: float
    '''
    file_ = open(file_, "a+")
    file_.writelines("Epoch %d step %d\n"%(epoch, step))
    file_.writelines("\tLoss average %f\n"%(loss_train))
    file_.writelines("\tLoss valid average %f, acc valid %f\n"%(loss_val, acc_val))
    file_.writelines("learning_rate %f\n"%(lr))

def len_train_datatset(dataset_dict, transform, split_train_val):
    '''
    target: get train_dataset from unsplit dataset
    input:
        - dataset_dict: Dictionary contain dataset information
        - transform 
        - split_train_val: ratio split
    '''
    DatasetClass = dataset_dict["class"]
    train_dataset = DatasetClass(dataset_dict["argument"],transform = transform, mode = "train")
    return len(train_dataset)*split_train_val



