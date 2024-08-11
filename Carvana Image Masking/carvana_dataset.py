import os
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

class CarvanaDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform

        # list data
        self.images = sorted(os.listdir(images_path))
        self.masks = sorted(os.listdir(masks_path))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: Any) -> (Image, Image):
        image_full_path = os.path.join(self.images_path, self.images[idx])
        mask_full_path = os.path.join(self.masks_path, self.masks[idx].replace(".jpg", ".png"))

        image = np.array(Image.open(image_full_path).convert("RGB"))
        # L --> Grayscale
        mask = np.array(Image.open(mask_full_path).convert("L"))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])


dataset = CarvanaDataset(
    images_path='train_images/',
    masks_path='train_masks/',
    transform=train_transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

for images, masks in train_loader:
    print(images.shape, masks.shape)
    break