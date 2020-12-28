import torchvision.transforms as transforms
from skimage import transform
import matplotlib.pyplot as plt
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h>w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img


    
if __name__ == "__main__":
    compose = transforms.Compose([
        Rescale(128)
    ])
    image = plt.imread("/home/huyen/data/bdd100k_seg/bdd100k/seg/color_labels/train/0a1d6940-9acd0000_train_color.png")
    new_img = compose(image)
    print(new_img.shape)
    plt.imshow(new_img)
    plt.show()