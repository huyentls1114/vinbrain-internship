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
lr_scheduler = {
    "class": ReduceLROnPlateau,
    "metric":"val_loss",
    "step_type":"epoch",
    "schedule_args":{
        "mode":"min",
        "factor":0.5,
        "patience":8,
        "threshold":1e-2,
        "min_lr":1e-6
    }
}
# net
from model.backbone import BackboneResnet18
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
num_classes = 1
net = {
    "class":UnetDynamic,
    "net_args":{
        "backbone_class": BackBoneResnet101Dynamic,
        "encoder_args":{
            "pretrained":True           
        },
        "decoder_args":{
            "img_size": image_size
        }
    }
}
net = {
    "class":UnetDynamic,
    "net_args":{
        "backbone_class": BackBoneResnet18Dynamic,
        "encoder_args":{
            "pretrained":True           
        },
        "decoder_args":{
            "img_size": image_size
        }
    }
}
net = {
    "class":Unet,
    "net_args":{
        "backbone_class": BackboneResnet18,
        "encoder_args":{
            "pretrained":True           
        },
        "decoder_args":{
            "bilinear": False,
            "pixel_shuffle":True
        }
    }
}
from model.backbone_resnet import BackboneResnet18
net = {
    "class":Unet,
    "net_args":{
        "backbone_class": BackboneResnet18,
        "encoder_args":{
            "pretrained":True           
        },
        "decoder_args":{
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
