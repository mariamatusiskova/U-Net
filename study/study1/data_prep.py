import os
import random
# Part of the OpenCV library, used for image processing tasks such as reading, writing, and manipulating images.
import cv2
# A library for handling arrays, used here to convert images to array format.
import numpy as np
import matplotlib.pyplot as plt
# A module from the Python Imaging Library (PIL), used for opening and manipulating images.
from PIL import Image

from config import *  # import basic paths

# loading the images and labels data
train_img_path = os.path.join(DATA_DIR, 'train-volume.tif')
train_label_path = os.path.join(DATA_DIR, 'train-labels.tif')

# hold the opened image and label files as PIL.Image objects. The TIFF files
# are expected to be multi-frame (contain multiple images in a single file)
train_imgs = Image.open(train_img_path)
train_labels = Image.open(train_label_path)

# Processing the Images and Labels
# An empty list to store tuples of image and label arrays.
data_list = list()

# train_imgs.n_frames: Retrieves the number of frames (images) in the TIFF file.
# iterates over each frame in the TIFF files
for n in range(train_imgs.n_frames):
    # Moves the file pointer to the nth frame in the TIFF file.
    train_imgs.seek(n)
    train_labels.seek(n)

    # Moves the file pointer to the nth frame in the TIFF file.
    train_img = np.asarray(train_imgs)
    train_label = np.asarray(train_labels)

    data_list.append((train_img, train_label))

# Shuffle and Split Data
random.shuffle(data_list)
train_list = data_list[:24]
val_list = data_list[24:27]
test_list = data_list[27:]

# Saving the Processed Data
for idx, data in enumerate(train_list, 1):
    img, label = data
    img_dst = os.path.join(TRAIN_IMGS_DIR, f'{idx:02}.png')
    label_dst = os.path.join(TRAIN_LABELS_DIR, f'{idx:02}.png')
    cv2.imwrite(img_dst, img)
    cv2.imwrite(label_dst, label)

for idx, data in enumerate(val_list, 1):
    img, label = data
    img_dst = os.path.join(VAL_IMGS_DIR, f'{idx:02}.png')
    label_dst = os.path.join(VAL_LABELS_DIR, f'{idx:02}.png')
    cv2.imwrite(img_dst, img)
    cv2.imwrite(label_dst, label)

for idx, data in enumerate(test_list, 1):
    img, label = data
    img_dst = os.path.join(TEST_IMGS_DIR, f'{idx:02}.png')
    label_dst = os.path.join(TEST_LABELS_DIR, f'{idx:02}.png')
    cv2.imwrite(img_dst, img)
    cv2.imwrite(label_dst, label)