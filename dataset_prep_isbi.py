import os

import cv2
import numpy as np
from PIL import Image
import random


def save_and_process_data(data_list: list, image_path: str, label_path: str):
    for index, data in enumerate(data_list, 1):
        image, label = data
        image_dataset = os.path.join(image_path, f'{index:02}.png')
        label_dataset = os.path.join(label_path, f'{index:02}.png')
        cv2.imwrite(image_dataset, image)
        cv2.imwrite(label_dataset, label)


train_volume_path = './ISBI 2012 challenge/train-volume.tif'
train_labels_path = './ISBI 2012 challenge/train-labels.tif'

test_volume_path = './ISBI 2012 challenge/test-volume.tif'
test_labels_path = './ISBI 2012 challenge/test-labels.tif'

train_images = Image.open(train_volume_path)
train_labels = Image.open(train_labels_path)

test_images = Image.open(test_volume_path)
test_labels = Image.open(test_labels_path)

train_data = list()
test_data = list()

for num in range(train_images.n_frames):
    train_images.seek(num)
    train_labels.seek(num)

    # converts the current frame (image or label) into a NumPy array
    train_image = np.array(train_images)
    train_label = np.array(train_labels)

    train_data.append((train_image, train_label))

# shuffle and Split Data
random.shuffle(train_data)
train_list = train_data[:24]
val_list = train_data[24:]

for num in range(test_images.n_frames):
    test_images.seek(num)
    test_labels.seek(num)

    # converts the current frame (image or label) into a NumPy array
    test_image = np.array(test_images)
    test_label = np.array(test_labels)

    test_data.append((test_image, test_label))

save_and_process_data(train_list, './data/train_img_data/', './data/train_label_data/')
save_and_process_data(val_list, './data/val_img_data/', './data/val_label_data/')
save_and_process_data(test_data, './data/test_img_data/', './data/test_label_data/')

