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
