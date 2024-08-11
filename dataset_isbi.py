import glob
import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class ISBIDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, transform=None):
        self.images =  sorted(glob.glob(os.path.join(images_dir, '*.png')))
        self.labels = sorted(glob.glob(os.path.join(labels_dir, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        # reading image at the index and normalizing the image and label
        # pixel to [0, 1]
        image = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE) / 255.
        label = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE) / 255.

        # temporary storage
        # shape (H, W, 1)  '1' for grayscale channel dimension
        return_dict = {
            'image': image[:, :, np.newaxis],
            'label': label[:, :, np.newaxis],
        }

        if self.transform:
            return_dict = self.transform(return_dict)

        return return_dict
