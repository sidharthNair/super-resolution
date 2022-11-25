import os
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import albumentations as A
import albumentations.pytorch as AT
from torch.utils.data import Dataset, DataLoader
from config import HR_SIZE, LR_SIZE

# For each image we take a randomly cropped and transformed image from the full image
randomize_crop = A.Compose([
    A.RandomCrop(width=HR_SIZE, height=HR_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])

# HR images are scaled to the range [-1, 1]
hr_transform = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    AT.ToTensorV2(),
])

# LR images are scaled to the range [0, 1]
lr_transform = A.Compose([
    A.Resize(width=LR_SIZE, height=LR_SIZE, interpolation=Image.Resampling.BICUBIC),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    AT.ToTensorV2(),
])

class CustomDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        self.data = []
        self.root = path
        self.classes = os.listdir(path)
        for i in range(len(self.classes)):
            files = os.listdir(os.path.join(path, self.classes[i]))
            self.data += list(zip(files, [i] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]
        dir = os.path.join(self.root, self.classes[label])

        img = np.array(Image.open(os.path.join(dir, file)))
        img = randomize_crop(image=img)["image"]
        lr = lr_transform(image=img)["image"]
        hr = hr_transform(image=img)["image"]
        return lr, hr
