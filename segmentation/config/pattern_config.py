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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
lr_scheduler_crf = {
    "class":OneCycleLR,
    "metric": None,
    "step_type":"iteration",
    "schedule_args":{
        "T0": 1,
        "Tmul":2
    }    
}
#OneCycleLR
from torch.optim.lr_scheduler import OneCycleLR
steps_per_epoch = int(len_train_datatset(dataset, transform_train, transform_label, 1)/batch_size)
num_epochs = 80
lr_scheduler = {
    "class":OneCycleLR,
    "metric": None,
    "step_type":"batch",
    "schedule_args":{
        "max_lr": 1e-4,
        "epochs":num_epochs,
        "steps_per_epoch":steps_per_epoch+1,
        "final_div_factor":10,
    }    
}

# net
num_classes = 1
from model.backbone import BackboneOriginal
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
num_classes = 1
net = {
    "class":Unet,
    "net_args":{
        "backbone_class": BackboneEfficientB0VGG,
        "encoder_args":{
            "pretrained":True           
        },
        "decoder_args":{
            "bilinear": False,
            "pixel_shuffle":True
        }
    }
}

from model.unet import UnetCRF
from model.backbone import BackboneResnet18VGG
num_classes = 1
current_epoch = 99
net = {
    "class":UnetCRF,
    "net_args":{
        "checkpoint_path": os.path.join(output_folder, "checkpoint_"+str(current_epoch))
        "backbone_class": BackboneResnet18VGG,
        "encoder_args":{
            "pretrained":True           
        },
        "decoder_args":{
            "pixel_shuffle":True,
            "bilinear":False
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

num_classes = 1
from loss.loss import DiceLoss
loss_function = {
    "class": DiceLoss,
    "loss_args":{
        "activation":nn.Sigmoid()
    }
}

from loss.loss import FocalLoss
loss_function = {
    "class": FocalLoss,
    "loss_args":{
        "alpha": 0.98,
        "gamma": 2
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
dataset = {
    "class": BrainTumorDataset,
    "dataset_args":{
        "input_folder":"/content/data/BrainTumor",
        "augmentation": A.Compose([
            A.Resize(512, 512),
            RandomCrop(450, 450),
            RandomVerticalFlip(p=0.5),
            RandomHorizontalFlip(p=0.5)
        ])
    }
}

dataset = {
    "class": BrainTumorDataset,
    "dataset_args":{
        "input_folder":"E:\data\BrainTumor",
        "augmentation": A.Compose([
            A.Resize(512, 512),
            RandomCrop(450, 450),
            RandomVerticalFlip(p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomRotate((0, 270), p = 0.5),
            RandomBlur(blur_limit = 10, p = 0.5)
        ])
    }
}

import albumentations as A
from dataset.transform import *
dataset = {
    "class": BrainTumorDataset,
    "dataset_args":{
        "input_folder":"/content/data/BrainTumor",
        "augmentation": A.Compose([
            A.Resize(512, 512),
            RandomCrop(450, 450, p = 0.5),
            A.OneOf([
                RandomVerticalFlip(p=0.5),
                RandomHorizontalFlip(p=0.5),
                RandomTranspose(p = 0.5),
            ]),
            RandomRotate((0, 270), p = 0.5),
            RandomBlur(blur_limit = 10, p = 0.1),
            CLAHE(p = 0.1),
            RandomBrightnessContrast(p = 0.1)
        ])
    }
}