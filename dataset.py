import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.mapping = {
               (0,0,0): 0,
               (31,120,180): 1,
               (227,26,28): 2,
               (106,61,154): 3}

    def __len__(self):
        return len(self.images)
        
    def mask_to_class(self, mask):
        for k in self.mapping:
           mask[mask==k] = self.mapping[k]
        return mask 

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        mask = self.mask_to_class(mask)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            mask = mask.permute(2,0,1)
           # print(f'{image.size()} and {mask.size()}')
            
        return image, mask

