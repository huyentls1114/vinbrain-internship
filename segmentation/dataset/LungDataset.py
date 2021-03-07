from torch.utils.data import Dataset
import os

class LungDataset(Dataset):
    def __init__(self, dataset_args, transform_image, transform_label, mode = "train"):
        super(LungDataset, self).__init__()
        if "augmentation" in dataset_args.keys():
            self.augmentation = dataset_args["augmentation"]
        else:
            self.augmentation = None


        self.covid_chesxray_folder = dataset_args["covid_chesxray_folder"]
        self.covid_chesxray_image_folder = os.path.join(self.covid_chesxray_folder, "images")
        self.covid_chesxray_mask_folder = os.path.join(self.covid_chesxray_folder, "masks")
        self.covid_chesxray_names = self.read_txt(self.covid_chesxray_folder, mode)

        self.transform_image = transform_image
        self.transform_label = transform_label

    def __len__(self):
        return len(self.covid_chesxray_names)
    
    def __getitem__(self, idx):
        img_name = self.covid_chesxray_names[idx]
        img_path = os.path.join(self.covid_chesxray_image_folder, img_name)
        image = plt.imread(img_path)
        # image = image[:, :, 0]

        mask_path = os.path.join(self.covid_chesxray_mask_folder, img_name)
        mask = plt.imread(mask_path)
        mask = mask[:, :, 0]

        if (self.mode == "train") and (self.augmentation is not None):
            # print(self.mode)
            augmented = self.augmentation(image = image, mask = mask)
            image, mask = augmented['image'], augmented['mask']
        return self.transform_image(np.array(image)), self.transform_label(np.array(mask))
        
    def load_sample(self, batch_size = 4):
        list_imgs = []
        list_masks = []

        #random list idx
        list_idx = np.arange(self.__len__())
        np.random.shuffle(list_idx)
        list_idx = list_idx[0:batch_size]

        for i in range(batch_size):
            image, mask = self.__getitem__(list_idx[i])
            list_imgs.append(image[None, :, :])
            list_masks.append(mask[None, :, :])
        return torch.cat(list_imgs), torch.cat(list_masks)

    def read_txt(self, input_folder, mode):
        if mode == "test":
            mode = "val"
        train_txt_file = os.path.join(input_folder, "%s.txt"%(mode))
        file_ = open(train_txt_file)
        list_ = file_.readlines()
        list_img_name = []
        for line in list_:
            img_name = line.replace("\n","")
            list_img_name.append(img_name)
        file_.close()
        return list_img_name