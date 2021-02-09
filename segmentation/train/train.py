import torch
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import numpy as np

from utils.utils import save_loss_to_file, compose_images
from visualize.visualize import Visualize
from model.unet import UnetCRF

class Trainer:
    def __init__(self, configs, data, copy_configs = True):
        self.data = data
        if "update_ds" in  configs.dataset["dataset_args"].keys():
            self.update_ds = configs.dataset["dataset_args"]["update_ds"]
        else:
            self.update_ds = None
        self.num_classes = configs.num_classes

        #net
        self.net = configs.net["class"](**configs.net["net_args"])

        #config cuda
        cuda = configs.device
        self.device = torch.device( cuda if cuda == "cpu" else "cuda:"+str(configs.gpu_id))
        self.net.to(self.device)

        #config train parameters
        self.batch_size = configs.batch_size
        self.num_epochs = configs.num_epochs

        #inititalize variables
        self.liss_loss = []
        self.current_epoch = 0

        #optimizer
         
        if configs.net["class"] == UnetCRF:
            self.lr = 1e-2
            lr_scheduler = configs.lr_scheduler_crf
            self.num_epochs = configs.num_epochs+10
            self.current_epoch = configs.current_epoch
        else:
            self.lr = configs.lr  
            lr_scheduler = configs.lr_scheduler 
        self.optimizer = configs.optimizer["class"](self.net.parameters(), self.lr, **configs.optimizer["optimizer_args"])
        self.initial_lr_scheduler(lr_scheduler)

        #loss and metric
        self.metric = configs.metric["class"](**configs.metric["metric_args"])
        if "metric_type" in configs.metric.keys():
            self.metric_type = configs.metric["metric_type"]
        else:
            self.metric_type = None
        self.crition = configs.loss_function["class"](**configs.loss_function["loss_args"])

        #files
        self.input_folder = configs.dataset["dataset_args"]["input_folder"]
        self.output_folder = configs.output_folder
        self.loss_file = configs.loss_file
        self.config_file_path = configs.config_file_path
        if copy_configs:
            self.initial_output_folder()

        #summary writer
        self.sumary_writer = SummaryWriter(self.output_folder)
        self.global_step = 0
        self.steps_per_epoch = len(self.data.train_loader)

        
        #initial Visualize
        self.image_size = configs.image_size
        self.visualize = Visualize(self.current_epoch, 
                                    self.num_epochs,
                                    self.data, 
                                    img_size = self.image_size)
        self.transform_test = configs.transform_test
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
                                    train_loss = self.visualize.train_loss,
                                    valid_loss = self.visualize.valid_loss, 
                                    img_size = self.image_size)
        for epoch in self.visualize.mb:
            if self.update_ds is not None:
                self.data.update_train_ds(**self.update_ds)
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint()                

    def train_one_epoch(self):
        train_loss = 0
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
                elif self.lr_schedule_step_type == "iteration":
                    self.schedule_lr(i)
            
            self.sumary_writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.sumary_writer.add_scalars('loss',{'train': loss.item()}, self.global_step)
            self.global_step+=1

        self.visualize_images()
        self.visualize_loss(train_loss, i+1)

    def visualize_loss(self, train_loss, num_batches, step = 0):
        i = step
        train_loss_avg = train_loss/num_batches
        val_loss_avg, val_acc_avg = self.evaluate(mode = "val", metric_type= self.metric_type)
        if self.lr_schedule_step_type == "epoch":
            self.schedule_lr(metric_value = eval(self.lr_scheduler_metric))
        lr = self.optimizer.param_groups[0]['lr']
        print("Epoch %3d step%3d: loss train: %5f, loss valid: %5f, dice valid: %5f, learning rate: %5f"%(self.current_epoch, i, train_loss_avg, val_loss_avg, val_acc_avg, lr))
        loss_file_path = os.path.join(self.output_folder, self.loss_file)
        save_loss_to_file(loss_file_path, self.current_epoch, i, train_loss_avg, val_loss_avg, val_acc_avg, lr)
        self.sumary_writer.add_scalars('loss', {'val': val_loss_avg}, self.global_step)
        self.sumary_writer.add_scalars('dice',{'val':val_acc_avg}, self.global_step)
        self.visualize.plot_loss_update(train_loss_avg, val_loss_avg)

    def visualize_images(self):
        images, labels = self.data.load_batch(mode = "train")
        images, labels = images.to(self.device), labels.to(self.device)
        val_imgs, val_labels = self.data.load_batch(mode = "val")
        val_imgs, val_labels = val_imgs.to(self.device), val_labels.to(self.device)
        with torch.no_grad():
            outputs = self.net(images)
            val_outputs = self.net(val_imgs)

        predicts = torch.sigmoid(outputs)
        self.sumary_writer.add_images("train/images", images, self.global_step)
        self.sumary_writer.add_images("train/mask", labels, self.global_step)
        self.sumary_writer.add_images("train/outputs", predicts, self.global_step)

        val_predicts = torch.sigmoid(val_outputs)
        self.sumary_writer.add_images("val/images", val_imgs, self.global_step)
        self.sumary_writer.add_images("val/mask", val_labels, self.global_step)
        self.sumary_writer.add_images("val/outputs", val_predicts, self.global_step)
        

        train_compose_images = compose_images(images[0], labels[0], predicts[0])
        val_compose_images = compose_images(val_imgs[0], val_labels[0], val_predicts[0])
        # import pdb; pdb.set_trace()
        self.visualize.update_image(np.vstack([train_compose_images, val_compose_images]))
    
    def evaluate(self, mode = "val", metric_type = "normal", metric = None):
        if metric is None:
            metric = self.metric
        progress = {
            "train":self.visualize.progress_train,
            "val":self.visualize.progress_val,
            "test":self.visualize.progress_test
        }
        metrict_list = []
        loss = 0
        self.net.eval()
        with torch.no_grad():
            for i, samples in enumerate(progress[mode]):
                images, labels = samples[0].to(self.device), samples[1].to(self.device)
                outputs = self.net(images)
                loss += self.crition(outputs, labels).item()
                metrict_list.append(metric(outputs, labels))
            metrict_list = torch.cat(metrict_list)
            return loss/(i+1), metrict_list.mean()
            
    def predict(self, img):
        self.net.eval()
        with torch.no_grad():
            img = img[:, :, 0]
            img_tensor = self.transform_test(img)
            img_tensor = img_tensor.to(self.device)
            img_tensor = img_tensor[None, :, :, :]
            output = self.net(img_tensor)
            predicts = torch.sigmoid(output)
            predicts = predicts[0].cpu().numpy().transpose(1, 2, 0)[:,:,0]
            return predicts

    def save_checkpoint(self, filename = None):
        if filename is None:
            filename = "checkpoint_%d"%(self.current_epoch)
        state_dict = {
            "current_epoch": self.current_epoch,
            "train_loss_list":self.visualize.train_loss,
            "val_loss_list":self.visualize.valid_loss,
            "net":self.net.state_dict(),
            "optimizer":self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "lr_scheduler_metric": self.lr_scheduler_metric,
            "lr_scheduler_step_type":self.lr_schedule_step_type,
        }
        filepath = os.path.join(self.output_folder, filename)
        torch.save(state_dict, filepath)

    def load_checkpoint(self, filename = None):
        if filename is None:
            filename = "checkpoint_%d"%(self.num_epochs-1)
        file_path = os.path.join(self.output_folder, filename)
        state_dict = torch.load(file_path, map_location=self.device)
        self.net.load_state_dict(state_dict["net"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.current_epoch = state_dict["current_epoch"]
        self.visualize.train_loss = state_dict["train_loss_list"]
        self.visualize.valid_loss = state_dict["val_loss_list"]
        if "lr_scheduler" in state_dict.keys():
            try:
                self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
            except:
                self.lr_scheduler = state_dict["lr_scheduler"]
            self.lr_scheduler_metric = state_dict["lr_scheduler_metric"]
            self.lr_schedule_step_type = state_dict["lr_scheduler_step_type"]

    def schedule_lr(self, iteration = 0, metric_value = 0):
        assert self.lr_scheduler is not None
        if self.lr_scheduler_metric is not None:
            if self.lr_schedule_step_type == "iteration":
                #for Cosine Anealing Warm Restart
                self.lr_scheduler.step(self.current_epoch+iteration/self.steps_per_epoch)
            else:
                #for ReduceLROnPlateau
                # val_loss, val_acc = self.evaluate(mode = "val")
                self.lr_scheduler.step(metric_value)
        else:
            self.lr_scheduler.step()

    def update(self, best_epoch = None, num_epochs = None, positive_rate = None, lr = None, lr_scheduler = None):
        if best_epoch is not None:
            # best_epoch = self.num_epochs - 1
            self.load_checkpoint("checkpoint_%d"%(best_epoch))
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if positive_rate is not None:
            self.update_ds["weight_positive"] = positive_rate
        if lr is not None:
            self.optimizer.param_groups[0]['lr'] = lr 
        if lr_scheduler is not None:
            self.initial_lr_scheduler(lr_scheduler)
