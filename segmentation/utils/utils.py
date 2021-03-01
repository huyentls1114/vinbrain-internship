import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def show_img(image):
    "show an image tensor"
    # image = (image/2)+0.5
    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()

def conver_numpy_image(image, normalize = False):
    '''
    convert tensor image to numpy array
    inputs:
        - image: torch tensor shape (C, H, W)
    '''
    image = image.cpu()
    if normalize:
        image = (image/2)+0.5
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image*255.0
    return image

def compose_images(image, mask, predict):
    '''
    concatenate images, masks, predict to 1 images
    inputs:
        - image, masks, predict: torch tensor shape (C, H, W)
    '''
    image = conver_numpy_image(image, normalize = True)
    mask = conver_numpy_image(mask, normalize = False)
    predict = conver_numpy_image(predict, normalize = False)
    if image.shape[2] == 3:
        image = image[:,:,0:1]
    import pdb; pdb.set_trace()
    if image.shape != predict.shape:
        image = cv2.resize(image, predict.shape[:2])
        mask = cv2.resize(mask, predict.shape[:2])
    return np.hstack([image, mask, predict])

def contour(image, mask):
    main = image.copy()
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i,c in enumerate(contours):
        colour = (1)
        cv2.drawContours(main,[c],-1,colour,2)
    return main

def save_loss_to_file(file_, epoch, loss_train, loss_val, acc_val, lr, step = 0):
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

def len_train_datatset(dataset_dict, transform_image, transform_label, split_train_val):
    '''
    target: get train_dataset from unsplit dataset
    input:
        - dataset_dict: Dictionary contain dataset information
        - transform 
        - split_train_val: ratio split
    '''
    DatasetClass = dataset_dict["class"]
    train_dataset = DatasetClass(dataset_dict["dataset_args"],transform_image = transform_image, transform_label= transform_label, mode = "train")
    return len(train_dataset)*split_train_val

def save_loss_to_file(file_, epoch, step, loss_train, loss_val, metric_val, lr):
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
    file_.writelines("Epoch %3d step%3d: loss train: %5f, loss valid: %5f, metric valid: %5f, learning rate: %5f"%(epoch, step, loss_train, loss_val, metric_val, lr))
    file_.writelines("\n")
    file_.close()

def caculate_num_parameter(net):
    return (p.numel() for p in net.parameters() if p.requires_grad)