import torch
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import numpy as np

from utils.utils import save_loss_to_file, compose_images
from visualize.visualize import Visualize

class Trainer:
    def __init__(self, configs, data):
        self.data = data
        self.num_classes = configs.net["net_args"]["output_channel"]

        #net
        self.net = configs.net["class"](**configs.net["net_args"])

        #config cuda
        cuda = configs.device
        self.device = torch.device( cuda if cuda == "cpu" else "cuda:"+str(configs.gpu_id))
        self.net.to(self.device)

        #optimizer
        self.lr = configs.lr    
        self.optimizer = configs.optimizer["class"](self.net.parameters(), self.lr, **configs.optimizer["optimizer_args"])
        self.initial_lr_scheduler(configs.lr_scheduler)

        #config train parameters
        self.batch_size = configs.batch_size
        self.num_epochs = configs.num_epochs
        self.steps_save_loss = configs.steps_save_loss
        self.steps_save_image = configs.steps_save_image
        
        #loss and metric
        self.metric = configs.metric["class"](**configs.metric["metric_args"])
        self.crition = configs.loss_function["class"](**configs.loss_function["loss_args"])

        #files
        self.input_folder = configs.dataset["dataset_args"]["input_folder"]
        self.output_folder = configs.output_folder
        self.loss_file = configs.loss_file
        self.config_file_path = configs.config_file_path
        self.initial_output_folder()

        #summary writer
        self.sumary_writer = SummaryWriter(self.output_folder)
        self.global_step = 0

        #inititalize variables
        self.liss_loss = []
        self.current_epoch = 0

        #initial Visualize
        self.image_size = configs.image_size
        
    
    def initial_lr_scheduler(self, lr_scheduler):
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler["class"](self.optimizer, **lr_scheduler["schedule_args"])
            self.lr_scheduler_metric = lr_scheduler["metric"]
            self.lr_schedule_step_type = lr_scheduler["step_type"]
        else:
            self.lr_scheduler = None
    
    def initial_output_folder(self):
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        shutil.copy(self.config_file_path, self.output_folder)

    def train(self, loss_file = None):
        if loss_file is not None:
            self.loss_file = loss_file
        self.visualize = Visualize(self.current_epoch, 
                                    self.num_epochs,
                                    self.data, 
                                    img_size = self.image_size)
        for epoch in self.visualize.mb:
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint()
            
            if (self.lr_scheduler is not None):
                if self.lr_schedule_step_type == "epoch":
                    self.schedule_lr()

    def train_one_epoch(self):
        train_loss = 0
        iteration = 0
        for i, sample in enumerate(self.visualize.progress_train):
            self.net.train()
            images, labels = sample[0].to(self.device), sample[1].to(self.device)
            outputs = self.net(images)
            # import pdb; pdb.set_trace()
            loss = self.crition(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            if self.lr_scheduler is not None:
                if self.lr_schedule_step_type == "batch":
                    self.schedule_lr()
            
            self.sumary_writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.sumary_writer.add_scalars('loss',{'train': loss.item()}, self.global_step)
            
            if iteration%(self.steps_save_image - 1) == 0:
                #summary image
                val_imgs, val_labels = self.data.load_batch(mode = "val")
                val_imgs, val_labels = val_imgs.to(self.device), val_labels.to(self.device)
                val_outputs = self.net(val_imgs)

                predicts = torch.sigmoid(outputs)>0.5
                self.sumary_writer.add_images("train/images", images, self.global_step)
                self.sumary_writer.add_images("train/mask", labels, self.global_step)
                self.sumary_writer.add_images("train/outputs", predicts, self.global_step)

                val_predicts = torch.sigmoid(val_outputs)>0.5
                self.sumary_writer.add_images("val/images", val_imgs, self.global_step)
                self.sumary_writer.add_images("val/mask", val_labels, self.global_step)
                self.sumary_writer.add_images("val/outputs", val_predicts, self.global_step)

                train_compose_images = compose_images(images[0], labels[0], predicts[0])
                val_compose_images = compose_images(val_imgs[0], val_labels[0], val_predicts[0])
                # import pdb; pdb.set_trace()
                self.visualize.update_image(np.vstack([train_compose_images, val_compose_images]))
            
            if iteration%(self.steps_save_loss - 1) == 0:
                train_loss_avg = train_loss/self.steps_save_loss
                val_loss_avg, val_acc_avg = self.evaluate(mode = "val")
                lr = self.optimizer.param_groups[0]['lr']
                print("Epoch %3d step%3d: loss train: %5f, loss valid: %5f, dice valid: %5f, learning rate: %5f"%(self.current_epoch, i, train_loss_avg, val_loss_avg, val_acc_avg, lr))
                train_loss = 0
                loss_file_path = os.path.join(self.output_folder, self.loss_file)
                save_loss_to_file(loss_file_path, self.current_epoch, i, train_loss_avg, val_loss_avg, val_acc_avg, lr)
                self.sumary_writer.add_scalars('loss', {'val': val_loss_avg}, self.global_step)
                self.sumary_writer.add_scalars('dice',{'val':val_acc_avg}, self.global_step)
                self.visualize.plot_loss_update(train_loss_avg, val_loss_avg)
            self.global_step+=1

    def evaluate(self, mode = "val", metric = None):
        if metric is None:
            metric = self.metric
        progress = {
            "train":self.visualize.progress_train,
            "val":self.visualize.progress_val,
            "test":self.visualize.progress_test
        }
        output_list = []
        label_list = []
        loss = 0
        self.net.eval()
        with torch.no_grad():
            for i, samples in enumerate(progress[mode]):
                images, labels = samples[0].to(self.device), samples[1].to(self.device)
                outputs = self.net(images)
                loss += self.crition(outputs, labels)
                if self.num_classes == 1:
                    outputs = torch.sigmoid(outputs)
                output_list.append(outputs)
                label_list.append(labels)
            output_tensor = torch.cat(output_list)
            label_tensor = torch.cat(label_list)
            metrict_result = metric(output_tensor, label_tensor)
            return loss/(i+1), metrict_result
            
    def predict(self, img):
        self.net.eval()
        with torch.no_grad():
            img_tensor = self.transform_test(img)
            img_tensor = img_tensor.to(self.device)
            output = self.net(img_tensor)
            return output

    def save_checkpoint(self, filename = None):
        if filename is None:
            filename = "checkpoint_%d"%(self.current_epoch)
        filepath = os.path.join(self.output_folder, filename)
        torch.save(self.net.state_dict(), filepath)

    def load_checkpoint(self, filename = None):
        if filename is None:
            filename = "checkpoint_%d"%(self.num_epochs-1)
        file_path = os.path.join(self.output_folder, filename)
        self.net.load_state_dict(torch.load(file_path, map_location=self.device))

    def schedule_lr(self, iteration = 0):
        assert self.lr_scheduler is not None
        if self.lr_scheduler_metric is not None:
            if self.lr_schedule_step_type == "iteration":
                #for Cosine Anealing Warm Restart
                self.lr_scheduler.step(self.current_epoch+iteration/self.batch_size)
            else:
                #for ReduceLROnPlateau
                val_loss, val_acc = self.evaluate(mode = "val")
                self.lr_scheduler.step(eval(self.lr_scheduler_metric))
        else:
            self.lr_scheduler.step()