import os
# For finding all file paths matching a specified pattern, such as all .png files in a directory.
import glob
import cv2
import numpy as np
import torch

__all__ = ['Dataset', 'ToTensor', 'GrayscaleNormalization', 'RandomFlip']


class Dataset(torch.utils.data.ISBIDataset):
    def __init__(self, imgs_dir, labels_dir, transform=None):
        self.transform = transform
        self.imgs = sorted(glob.glob(os.path.join(imgs_dir, '*.png')))
        self.labels = sorted(glob.glob(os.path.join(labels_dir, '*.png')))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Reads the image and label in grayscale and normalizes pixel values by dividing by 255.
        img = cv2.imread(self.imgs[index], cv2.IMREAD_GRAYSCALE) / 255.
        label = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE) / 255.

        # Adds a channel dimension to the image and label, converting them to shape (H, W, 1).
        ret = {
            'img': img[:, :, np.newaxis],
            'label': label[:, :, np.newaxis],
        }

        if self.transform:
            ret = self.transform(ret)

        return ret

# Converts image and label data from NumPy arrays to PyTorch tensors.
class ToTensor:
    def __call__(self, data):
        img, label = data['img'], data['label']

        # Rearranges the dimensions of the image and label
        # arrays from (H, W, C) to (C, H, W) to match PyTorch's expected input format.
        img = img.transpose(2, 0, 1).astype(np.float32)
        label = label.transpose(2, 0, 1).astype(np.float32)

        # Converts NumPy arrays to PyTorch tensors
        ret = {
            'img': torch.from_numpy(img),
            'label': torch.from_numpy(label),
        }
        # Returns a dictionary containing the tensor-converted image and label.
        return ret


# Defining the GrayscaleNormalization Transformation
class GrayscaleNormalization:
    #  Initializes the mean and standard deviation values for normalization.
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        img, label = data['img'], data['label']
        # Applies normalization to the image data.
        img = (img - self.mean) / self.std

        ret = {
            'img': img,
            'label': label,
        }
        # Returns the normalized image along with the unchanged label.
        return ret

# Defining the RandomFlip Transformation
# Randomly flips the image and label horizontally and/or vertically.
class RandomFlip:
    def __call__(self, data):
        img, label = data['img'], data['label']

        # Generates a random number and flips the image and label if the number is greater than 0.5.
        if np.random.rand() > 0.5:
            # Flips the image horizontally (left-right).
            img = np.fliplr(img)
            label = np.fliplr(label)

        if np.random.rand() > 0.5:
            # Flips the image vertically (up-down).
            img = np.flipud(img)
            label = np.flipud(label)

        ret = {
            'img': img,
            'label': label,
        }
        # Flips the image vertically (up-down).
        return ret