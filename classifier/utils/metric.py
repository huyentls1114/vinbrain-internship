import numpy as np
import torch

def abs(outputs, labels):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if not isinstance(outputs[0], int):
        predicted = np.argmax(outputs, 1)
    else:
        predicted = outputs
    return (predicted == labels).sum().item()