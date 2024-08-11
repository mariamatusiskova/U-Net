import numpy as np
import torch


class ToTensor:
    def __call__(self, data: dict) -> dict:
        image, label = data['image'], data['label']

        # rearranges the dimensions of the image and label
        # arrays from (H - 0, W - 1, C - 2) to (C, H, W) to match PyTorch's expected input format.
        image = image.transpose(2, 0, 1).astype(np.float32)
        label = label.transpose(2, 0, 1).astype(np.float32)

        return_dict = {
            # 'image': image[:, :, np.newaxis],
            # 'label': label[:, :, np.newaxis],
            # 'image': image,
            # 'label': label,
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(label),
        }

        return return_dict
