import torch
import os
from shutil import copy
from utils.utils import save_loss_to_file
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import ListDataset

class Trainer:
    def __init__(self, configs, data):
        '''
        target: initialize trainer for training
        inputs:
            - configs contains parameter for training: 
                - lr, batch_size, num_epoch, steps_save_loss, output_folder, device
                - loss_function
                - net: dict - contrain information of model
                - optimizer: dict - contain information of optimizer
                - transform: use for predict list image
                - lr_schedule: dict - contain information for schedule learning rate
                - metric: dict - information of metric for valid and test
                - loss_file: String - name of file in output_folder contain loss training process
            - data: instance Data classes in data folder
        '''
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.num_epochs = configs.num_epochs
        self.crition = configs.loss_function()
        self.net = configs.net["class"](**configs.net["net_args"])
        self.optimizer = configs.optimizer["class"](self.net.parameters(), self.lr, **configs.optimizer["optimizer_args"])
        self.transform_test = configs.transform_test

        #schedule learning rate
        if configs.lr_schedule is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = configs.lr_schedule["class"](self.optimizer, **configs.lr_schedule["schedule_args"])
            self.lr_shedule_metric = configs.lr_schedule["metric"]
            self.lr_schedule_step_type = configs.lr_schedule["step_type"]

        #data
        self.data = data

        #evaluate
        self.metric = configs.metric["class"](**configs.metric["metric_args"])

        #training process
        self.current_epoch = 0
        self.list_loss = []
        self.steps_save_loss = configs.steps_save_loss
        self.output_folder = configs.output_folder
        self.config_files = configs.config_files

        #define loss file
        self.loss_file = configs.loss_file

        #config cuda
        cuda = configs.device
        self.device = torch.device(cuda if cuda == "cpu" else "cuda:"+str(configs.gpu_id))
        self.net.to(self.device)

        #config output
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        copy(self.config_files, self.output_folder)

        #tensorboard
        self.summaryWriter = SummaryWriter(self.output_folder)
        self.global_step = 0

    def train(self, loss_file = None):
        '''
        target: training the model
        input:
            - loss_file: file contain loss of training process
        '''
        if loss_file is not None:
            self.loss_file = loss_file        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint()
            
            if self.lr_scheduler is not None:
                if self.lr_schedule_step_type == "epoch":
                    self.schedule_lr()

    def test(self):
        '''
        target: test the model
        '''
        loss, acc = self.evaluate(test)
        print("Test loss: %f test acc %f"%(loss, acc))
    
    def train_one_epoch(self):
        '''
        target: train per epoch
            - load image form train loader
            - train
            - save train result to summary writer
            - update learning rate if necessary
        '''
        self.net.train()
        train_loss = 0
        for i, sample in enumerate(self.data.train_loader):
            images, labels = sample[0].to(self.device), sample[1].to(self.device)
            outputs = self.net(images)
            loss = self.crition(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            if self.lr_scheduler is not None:
                if self.lr_schedule_step_type == "batch":
                    self.schedule_lr(i)
            self.summaryWriter.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.summaryWriter.add_scalars('loss',
                                            {
                                                'loss_train': loss.item()
                                            }, self.global_step)
            if i% (self.steps_save_loss-1) == 0:
                print("Epoch %d step %d"%(self.current_epoch, i))
                train_loss_avg = train_loss/self.steps_save_loss
                print("\tLoss average %f"%(train_loss_avg))
                val_loss_avg, val_acc_avg = self.evaluate(mode = "val")
                print("\tLoss valid average %f, acc valid %f"%(val_loss_avg, val_acc_avg))
                print("learning_rate ", self.optimizer.param_groups[0]['lr'])
                train_loss = 0.0
                loss_file_path = os.path.join(self.output_folder, self.loss_file)
                save_loss_to_file(loss_file_path, self.current_epoch, i, train_loss_avg, val_loss_avg, val_acc_avg, self.optimizer.param_groups[0]['lr'])
                self.summaryWriter.add_scalars('loss',{'loss_val': val_loss_avg}, self.global_step)
                self.summaryWriter.add_scalars('acc',{'acc_val': val_acc_avg}, self.global_step)
            self.global_step +=1
    def schedule_lr(self, iteration = 0):
        '''
        target: update learning rate schedule
        input:
            -iteration: Interger - iteration of each epoch, using for mode batch
        '''
        if not self.lr_scheduler is None:
            if self.lr_shedule_metric is not None:
                if self.lr_shedule_metric == "epoch":
                     self.lr_scheduler.step(self.current_epoch+iteration/self.batch_size)
                else:
                    val_loss, val_acc = self.evaluate(mode = "val")
                    self.lr_scheduler.step(eval(self.lr_shedule_metric))
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
        loader = {
            "val": self.data.val_loader,
            "train": self.data.train_loader,
            "test": self.data.test_loader
        }
        output_list = []
        label_list = []
        loss = 0
        self.net.eval()
        with torch.no_grad():
            for i, samples in enumerate(loader[mode]):
                images, labels = samples[0].to(self.device), samples[1].to(self.device)
                outputs = self.net(images)
                loss += self.crition(outputs, labels)
                output_list.append(outputs)
                label_list.append(labels)
            # import pdb; pdb.set_trace()
            output_tensor = torch.cat(output_list)
            label_tensor =  torch.cat(label_list)
            metric_result = metric(output_tensor, label_tensor)
            return loss/(i+1), metric_result

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
        '''
        target: save checkpoint to file
        input:
            - filename: String - file name to save checkpoint
        '''
        if filename is None:
            filename = "checkpoint_%d"%(self.current_epoch)
        file_path = os.path.join(self.output_folder, filename)
        torch.save(self.net.state_dict(), file_path)
    
    def load_checkpoint(self, filename = None):
        '''
        target: load checkpoint from file
        input:
            - filename: String - file name to load checkpoint
        '''
        if filename is None:
            filename = "checkpoint_%d"%(self.num_epochs-1)
        file_path = os.path.join(self.output_folder, filename)
        self.net.load_state_dict(torch.load(file_path, map_location=self.device))
