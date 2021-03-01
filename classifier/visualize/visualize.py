from IPython.display import display
from PIL import Image
from fastprogress.fastprogress import master_bar, progress_bar
from time import sleep
import numpy as np
import random
import matplotlib.pyplot as plt

class Visualize:
    def __init__(self, current_epoch, epochs, data, img_size = 256, train_loss = [], valid_loss = []):
        self.mb = master_bar(range(current_epoch, epochs))
        self.progress_train = progress_bar(data.train_loader,parent=self.mb)
        self.progress_val = progress_bar(data.val_loader)
        self.progress_test = progress_bar(data.test_loader)
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.mb.imgs_out = None

    def update(self, current_epoch = None, epochs = None, data = None, img_size = None, train_loss = None, valid_loss = None):
        if current_epoch is not None:
            assert epochs is not None
            assert data is not None
            self.mb = master_bar(range(current_epoch, epochs))
            self.progress_train = progress_bar(data.train_loader,parent=self.mb)
            self.mb.imgs_out = None
        if data is not None:
            self.progress_train = progress_bar(data.train_loader,parent=self.mb)
        if train_loss is not None:
            self.train_loss = train_loss
        if valid_loss is not None:
            self.valid_loss = valid_loss

    def update_image(self, img):
        '''
        input: numpy array image
        '''
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3 and (img.shape[2] == 1):
                img = img[:,:,0]
            img = Image.fromarray(img)
            img = img.convert('RGB')
        # import pdb; pdb.set_trace()
        if self.mb.imgs_out is None:
            self.mb.imgs_out = display(img, display_id=True)
        else:
            self.mb.imgs_out.update(img)

    def plot_loss_update(self, train_loss, valid_loss):
        # print(type(train_loss), type(valid_loss))
        x = range(len(self.train_loss)+1)
        self.train_loss.append(train_loss)
        self.valid_loss.append(valid_loss)
        y = np.concatenate((self.train_loss, self.valid_loss))
        graphs = [[x, self.train_loss], [x, self.valid_loss]]
        x_margin = 0.1
        y_margin = 0.05
        x_bounds = [np.min(x) - x_margin, np.max(x) + x_margin]
        y_bounds = [np.min(y) - y_margin, np.max(y) + y_margin]
        # import pdb; pdb.set_trace()
        self.mb.update_graph(graphs, x_bounds, y_bounds)
