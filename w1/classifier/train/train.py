import torch
import os
from shutil import copy

class Trainer:
    def __init__(self, configs, data):
        self.lr = configs.lr
        self.num_epochs = configs.num_epochs
        self.crition = configs.loss_function()
        self.net = configs.net()
        self.optimizer = configs.optimizer["class"](self.net, self.lr, configs.optimizer["optimizer_args"])
        self.data = data

        self.current_epoch = 0
        self.list_loss = []
        self.steps_save_loss = 2000
        self.output_folder = configs.output_folder
        self.config_files = configs.config_files

        cuda = configs.device
        self.device = torch.device(cuda if cuda == "cpu" else "cuda:"+str(configs.gpu_id))
        self.net.to(self.device)

    
    def train(self):
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        copy(self.config_files, self.output_folder)
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint()
    def test(self):
        loss, acc = self.evaluate(test)
        print("Test loss: %f test acc %f"%(loss, acc))
    def train_one_epoch(self):
        train_loss = 0
        for i, sample in enumerate(self.data.train_loader):
            images, labels = sample
            outputs = self.net(images)
            loss = self.crition(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            if i% (self.steps_save_loss-1) == 0:
                print("Epoch %d step %d"%(self.current_epoch, i))
                train_loss_avg = train_loss/self.steps_save_loss
                print("\tLoss average %f"%(train_loss_avg))
                val_loss_avg, val_acc_avg = self.evaluate(mode = "val")
                print("\tLoss valid average %f, acc valid %f"%(val_loss_avg, val_acc_avg))
                train_loss = 0.0
    
    def evaluate(self, mode = "val"):
        loader = {
            "val": self.data.val_loader,
            "train": self.data.train_loader,
            "test": self.data.test_loader
        }
        num_total = 0
        num_correct = 0
        loss = 0
        with torch.no_grad():
            for i, samples in enumerate(loader[mode]):
                images, labels = samples[0].to(self.device), samples[1].to(self.device)
                outputs = self.net(images)
                loss += self.crition(outputs, labels)
                num_total += outputs.size(0)
                num_correct += self.num_correct(outputs, labels)
            return loss/(i+1), num_correct/num_total
    def num_correct(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        return (predicted == labels).sum().item()
                
    def save_checkpoint(self, filename = None):
        if filename is None:
            filename = "checkpoint_%d"%(self.current_epoch)
        file_path = os.path.join(self.output_folder, filename)
        torch.save(self.net.state_dict(), file_path)