import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import utils
import PIL.Image as Image

class Dataset(data.Dataset):
    def __init__(self, path):
        super().__init__()

        self.data = []
        self.root = path
        self.classes = os.listdir(path)

        for i in range(self.classes):
            files = os.listdir(os.path.join(path, self.classes[i]))
            self.data += list(zip(files, [i] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]
        dir = os.path.join(self.root, self.classes[label])

        img = np.array(Image.open(os.path.join(dir, file)))
