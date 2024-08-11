import matplotlib.pyplot as plt
import cv2
import numpy as np

train_images_path = '../Carvana Image Masking/train_images/0cdf5b5d0ce1_01.jpg'
train_masks_path = '../Carvana Image Masking/train_masks/0cdf5b5d0ce1_01.png'

image = cv2.imread(train_images_path, cv2.IMREAD_GRAYSCALE)
coloured_image = cv2.imread(train_images_path)
mask = cv2.imread(train_masks_path, cv2.IMREAD_GRAYSCALE)
coloured_mask_rounded = cv2.imread(train_masks_path, cv2.IMREAD_GRAYSCALE).round()

figure, _ = plt.subplots(nrows=2, ncols=2)

figure.axes[0].imshow(coloured_image)
figure.axes[1].imshow(coloured_mask_rounded)

figure.axes[2].imshow(image, cmap='gray')
figure.axes[3].imshow(mask, cmap='gray')

plt.show()