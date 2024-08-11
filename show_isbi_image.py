# for reading and writing TIFF (Tagged Image File Format)
import tifffile as tiff
from skimage.transform import resize
import matplotlib.pyplot as plt

train_volume = tiff.imread('./ISBI 2012 challenge/train-volume.tif')
train_labels = tiff.imread('./ISBI 2012 challenge//train-labels.tif')

print(train_volume.shape)
print(train_labels.shape)

resized_image = resize(train_volume[0], (128, 128))
resized_mask = resize(train_labels[0], (128, 128))
resized_mask_rounded = resize(train_labels[0], (128, 128)).round()

figure, _ = plt.subplots(nrows=2, ncols=2)
figure.axes[0].imshow(resized_image)
figure.axes[1].imshow(resized_mask)

figure.axes[2].imshow(resized_image, cmap='gray')
figure.axes[3].imshow(resized_mask_rounded, cmap='gray')
plt.show()
