# randomly flips the image and label horizontally and/or vertically.
import random

import numpy as np

# mirroring????
class RandomFlip:
    def __call__(self, data: dict) -> dict:
        image, label = data['image'], data['label']

        # Generates a random number and flips the image and label if the number is greater than 0.5.
        if np.random.rand() > 0.5:
            # Flips the image horizontally (left-right).
            image = np.fliplr(image)
            label = np.fliplr(label)

        if np.random.rand() > 0.5:
            # Flips the image vertically (up-down).
            image = np.flipud(image)
            label = np.flipud(label)

        return_dict = {
            'image': image,
            'label': label,
        }

        return return_dict

class RandomRotate:
    def __call__(self, data: dict) -> dict:
        image, label = data['image'], data['label']
        angle = random.choice([0, 90, 180, 270])
        image = np.rot90(image, k=angle // 90)
        label = np.rot90(label, k=angle // 90)
        return {'image': image, 'label': label}