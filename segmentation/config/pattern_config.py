#lr schedule
lr_scheduler = {
    "class": ReduceLROnPlateau,
    "metric":"val_loss",
    "step_type":"epoch",
    "schedule_args":{
        "mode":"min",
        "factor":0.5,
        "patience":4,
        "threshold":1e-2,
        "min_lr":1-5
    }
}

# net
net = {
    "class":Unet,
    "net_args":{
        "backbone_class": BackboneOriginal,
        "encoder_args":{},
        "decoder_args":{
            "bilinear": True
        }
    }
}

#metric
metric = {
    "class":Dice_Score,
    "metric_args":{
        "threshold":0.5,
        "epsilon":1e-4
    }
}

#loss function
loss_function = {
    "class": nn.BCEWithLogitsLoss,
    "loss_args":{
    }
}


#optimizer
optimizer = {
    "class":Adam,
    "optimizer_args":{
    }
}

#dataset
dataset = {
    "class": BrainTumorDataset,
    "dataset_args":{
        "input_folder":"E:\data\BrainTumor"
    }
}
