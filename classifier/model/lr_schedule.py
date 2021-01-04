#file contain template that declare lr_schedule in config
lr_schedule = {
    "class":CyclicLR,
    "metric":None,
    "step_type":"epoch",
    "optimizer_args":{
        "epochs":num_epochs,
        "max_lr":0.3,
        "steps_per_epoch":int(train_length/batch_size)
    }
}

#MultistepLR
from torch.optim.lr_scheduler import MultiStepLR
configs.num_epochs = 20
configs.lr_schedule ={
    "class": MultiStepLR,
    "metric":None,
    "step_type":"epoch",
    "schedule_args":{
        "milestones":[10, 15],
        "gamma":0.1,
    }
}

#MultistepLR
from torch.optim.lr_scheduler import MultiStepLR
configs.num_epochs = 20
configs.lr_schedule ={
    "class": MultiStepLR,
    "metric":None,
    "step_type":"epoch",
    "schedule_args":{
        "milestones":[10, 15],
        "gamma":0.1,
    }
}
trainer = Trainer(configs, data)
trainer.train(loss_file = "loss_file_multisteplr.txt")

#ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau
configs.num_epochs = 30
configs.lr_schedule={
    "class":ReduceLROnPlateau,
    "metric":"val_loss",
    "step_type":"epoch",
    "schedule_args":{
        "mode":"min",
        "factor":0.1,
        "patience":4,
        "threshold":1e-2,
        "min_lr":1e-4
    }
}
trainer = Trainer(configs, data)
trainer.train("loss_file_ReduceLROnPlateau.txt")

#OneCycleLR
from torch.optim.lr_scheduler import OneCycleLR
configs.num_epochs = 30
configs.lr_schedule = {
    "class":OneCycleLR,
    "metric": None,
    "step_type":"batch",
    "schedule_args":{
        "max_lr":0.01,
        "epochs":configs.num_epochs,
        "steps_per_epoch":configs.steps_per_epoch+1
    }    
}
trainer = Trainer(configs, data)
trainer.train(loss_file = "loss_file_OneCycleLR.txt")

#CosinAnealingWarmRestart
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
configs.num_epochs = 30
configs.lr_schedule = {
    "class": CosineAnnealingWarmRestarts,
    "metric":"epoch",
    "step_type":"batch",
    "schedule_args":{
        "T_0":configs.batch_size
    }
}
trainer = Trainer(configs, data)
trainer.train("file_loss_CosineAnnealingWarmRestarts.txt")