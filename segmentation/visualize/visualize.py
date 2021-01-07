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

    def example(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = 0.5 - 0.06 * epoch + random.uniform(0, 0.04)
            valid_loss = 0.5 - 0.03 * epoch + random.uniform(0, 0.04)
            self.plot_loss_update(train_loss, valid_loss)

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
        train_loss = train_loss.cpu()
        valid_loss = valid_loss.cpu()
        x = range(len(self.train_loss)+1)
        self.train_loss.append(train_loss)
        self.valid_loss.append(valid_loss)
        y = np.concatenate((self.train_loss, self.valid_loss))
        graphs = [[x, self.train_loss], [x, self.valid_loss]]
        x_margin = 0.1
        y_margin = 0.05
        x_bounds = [np.min(x) - x_margin, np.max(x) + x_margin]
        y_bounds = [np.min(y) - y_margin, np.max(y) + y_margin]
        self.mb.update_graph(graphs, x_bounds, y_bounds)
