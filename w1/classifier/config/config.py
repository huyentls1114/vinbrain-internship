import torchvision.transforms as transforms
from utils.transform import Rescale


batch_size = 4
split_train_val = 0.7
lr = 0.01
n_gpu = 0.01
classes = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                        ])
