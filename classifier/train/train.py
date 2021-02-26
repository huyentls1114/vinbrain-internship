import torch
import os
from shutil import copy
from utils.utils import save_loss_to_file
import shutil
from torch.utils.tensorboard import SummaryWriter
# from dataset.dataset import ListDataset
from visualize.visualize import Visualize
class Trainer:
    def __init__(self, configs, data, copy_configs = True):
        '''
        target: initialize trainer for training
        inputs:
            - configs contains parameter for training: 
                - lr, batch_size, num_epoch, steps_save_loss, output_folder, device
                - loss_function
                - net: dict - contrain information of model
                - optimizer: dict - contain information of optimizer
                - transform: use for predict list image
                - lr_scheduler: dict - contain information for schedule learning rate
                - metric: dict - information of metric for valid and test
                - loss_file: String - name of file in output_folder contain loss training process
            - data: instance Data classes in data folder
        '''
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.num_epochs = configs.num_epochs

        #data
        self.data = data
        

        #loss, model, optimizer, metric 
        self.crition = configs.loss["class"](**configs.loss["loss_args"])
        self.net = configs.net["class"](**configs.net["net_args"])
        self.optimizer = configs.optimizer["class"](self.net.parameters(), self.lr, **configs.optimizer["optimizer_args"])
        self.metric = configs.metric["class"](**configs.metric["metric_args"])

        #scheduler
        self.init_lr_scheduler(configs.lr_scheduler)

        #training process
        self.current_epoch = 0
        self.list_loss = []
        self.output_folder = configs.output_folder
        self.config_file_path = configs.config_file_path
        #config output
        if copy_configs:
            self.initial_output_folder()

        #visualize
        self.image_size = configs.img_size
        self.visualize = Visualize(self.current_epoch, 
                                    self.num_epochs,
                                    self.data, 
                                    img_size = self.image_size)
        self.loss_file = configs.loss_file



        #config cuda
        cuda = configs.device
        self.device = torch.device(cuda if cuda == "cpu" else "cuda:"+str(configs.gpu_id))
        self.net.to(self.device)

        #tensorboard
        self.sumary_writer = SummaryWriter(self.output_folder)
        self.global_step = 0

        #test image
        self.transform_test = configs.transform_test

        self.steps_per_epoch = configs.steps_per_epoch
    def initial_output_folder(self):
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        shutil.copy(self.config_file_path, self.output_folder)

    def init_lr_scheduler(self, lr_scheduler):
        #schedule learning rate
        if lr_scheduler is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = lr_scheduler["class"](self.optimizer, **lr_scheduler["schedule_args"])
            self.lr_scheduler_metric = lr_scheduler["metric"]
            self.lr_scheduler_step_type = lr_scheduler["step_type"]
            

    def train(self, loss_file = None):
        '''
        target: training the model
        input:
            - loss_file: file contain loss of training process
        '''
        self.visualize.update(current_epoch = self.current_epoch,
                              epochs = self.num_epochs,
                              data = self.data)
        if loss_file is not None:
            self.loss_file = loss_file        
        for epoch in self.visualize.mb:
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint()
            if self.lr_scheduler is not None:
                if (self.lr_scheduler_step_type == "epoch") and (self.lr_scheduler_metric is None):
                    self.schedule_lr()

    def test(self):
        '''
        target: test the model
        '''
        loss, acc = self.evaluate("test")
        print("Test loss: %f test acc %f"%(loss, acc))
    
    def train_one_epoch(self):
        '''
        target: train per epoch
            - load image form train loader
            - train
            - save train result to summary writer
            - update learning rate if necessary
        '''
        
        train_loss = 0
        for i, sample in enumerate(self.visualize.progress_train):
            self.net.train()
            images, labels = sample[0].to(self.device), sample[1].to(self.device)
            outputs = self.net(images)
            loss = self.crition(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss_batch = loss.item()
            train_loss += train_loss_batch
            if self.lr_scheduler is not None:
                if self.lr_scheduler_step_type == "iteration":
                    self.schedule_lr(iteration = i)
                elif self.lr_scheduler_step_type == "batch":
                    self.schedule_lr()
            self.sumary_writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.sumary_writer.add_scalars('loss',{'train': loss.item()}, self.global_step)
            self.global_step+=1
            # if i%100 == 0:
            self.visualize_loss(train_loss_batch, 1)

    def visualize_loss(self, train_loss, num_batches, step = 0):
        i = step
        train_loss_avg = train_loss/num_batches
        val_loss_avg, val_acc_avg = self.evaluate(mode = "val")
        
        if (self.lr_scheduler_step_type == "epoch") and (self.lr_scheduler_metric is not None):
            self.schedule_lr(metric_value = eval(self.lr_scheduler_metric))

        lr = self.optimizer.param_groups[0]['lr']
        print("Epoch %3d step%3d: loss train: %5f, loss valid: %5f, acc valid: %5f, learning rate: %5f"%(self.current_epoch, i, train_loss_avg, val_loss_avg, val_acc_avg, lr))
        loss_file_path = os.path.join(self.output_folder, self.loss_file)
        save_loss_to_file(loss_file_path, self.current_epoch, i, train_loss_avg, val_loss_avg, val_acc_avg, lr)
        self.sumary_writer.add_scalars('loss', {'val': val_loss_avg}, self.global_step)
        self.sumary_writer.add_scalars('acc',{'val':val_acc_avg}, self.global_step)
        self.visualize.plot_loss_update(train_loss_avg, val_loss_avg.cpu().numpy())

    def schedule_lr(self, iteration = None, metric_value = None):
        assert self.lr_scheduler is not None
        # print("iteration", iteration, "metric_value", metric_value)
        if iteration is not None:
            #for Cosine Anealing Warm Restart
            self.lr_scheduler.step(self.current_epoch+iteration/self.steps_per_epoch)
        elif metric_value is not None:
            #for ReduceLROnPlateau
            # val_loss, val_acc = self.evaluate(mode = "val")
            self.lr_scheduler.step(metric_value)
        else:
            self.lr_scheduler.step()
            
    def evaluate(self, mode = "val", metric = None):
        '''
        target: caculate model with given metric
        input:
            - mode: String - ["train", "val", "test"]
            - metric: class of metric
        output:
            - loss: average loss of whole dataset
            - metric_value
        '''
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
                loss += self.crition(outputs, labels)
                metrict_list.append(metric(outputs, labels))
            metrict_list = torch.stack(metrict_list)
            return loss/(i+1), metrict_list.mean()

    def get_prediction(self, list_img):
        '''
        targets: get output of model from given list of images
        inputs:
            - list_img: list of image
        output: list of outputs model
        '''
        dataset = ListDataset(list_img, self.transform_test)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle= False, batch_size = self.batch_size)
        self.net.eval()
        output_list = []
        with torch.no_grad():
            for i, images in enumerate(dataloader):
                images = images.to(self.device)
                outputs = self.net(images)
                output_list.append(outputs)
        return torch.cat(output_list)

    def predict(self, img):
        '''
        targets: get output of model from given 1 image
        inputs:
            - img: Image from image io
        output: list of outputs model
        '''
        self.net.eval()
        with torch.no_grad():
            img_tensor = self.transform_test(img)
            img_tensor = img_tensor.to(self.device)
            output = self.net(img_tensor)
            return output


    def num_correct(self, outputs, labels):
        '''
        target: calculate number of element true
        '''
        _, predicted = torch.max(outputs, 1)
        return (predicted == labels).sum().item()
                
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
            "lr_scheduler_step_type":self.lr_scheduler_step_type,
        }
        filepath = os.path.join(self.output_folder, filename)
        torch.save(state_dict, filepath)
    
    def load_checkpoint(self, filename = None, file_path = None):
        '''
        target: load checkpoint from file
        input:
            - filename: String - file name to load checkpoint
        '''
        if filename is None:
            filename = "checkpoint_%d"%(self.num_epochs-1)
        if file_path is None:
            file_path = os.path.join(self.output_folder, filename)
        state_dict = torch.load(file_path, map_location=self.device)
        file_path = os.path.join(self.output_folder, filename)

        state_dict = torch.load(file_path, map_location=self.device)
        self.net.load_state_dict(state_dict["net"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.current_epoch = state_dict["current_epoch"]+1
        self.visualize.update(current_epoch = self.current_epoch,
                              epochs = self.num_epochs,
                              data = self.data,
                              train_loss = state_dict["train_loss_list"],
                              valid_loss = state_dict["val_loss_list"])
