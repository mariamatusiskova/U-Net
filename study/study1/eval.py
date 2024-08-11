import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms
from dataset import *
from torch.utils.data import DataLoader
from model import UNet
from utils import *
from config import *

# Configuration and Transformation
# Loads the configuration settings, like learning rate and batch size.
cfg = Config()
# Composes a series of transformations
# GrayscaleNormalization: Normalizes the grayscale image.
# ToTensor: Converts the image and label data from NumPy arrays to PyTorch tensors.
transform = transforms.Compose([
    GrayscaleNormalization(mean=0.5, std=0.5),
    ToTensor(),
])

# Setting Up Directories
# Directory where the evaluation results (images) will be saved. The directory is created if it doesn’t exist.
RESULTS_DIR = os.path.join(ROOT_DIR, 'test_results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Dataset and DataLoader
test_dataset = Dataset(imgs_dir=TEST_IMGS_DIR, labels_dir=TEST_LABELS_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

# Total number of samples in the test dataset.
test_data_num = len(test_dataset)
# The total number of batches for testing, calculated by dividing the number of test samples by the batch size.
test_batch_num = int(np.ceil(test_data_num / cfg.BATCH_SIZE))

# Model Setup
# Network
# Initializes the U-Net model and moves it to the GPU (or CPU) as defined by device.
net = UNet().to(device)

# Loss Function
# Sets up the loss function, Binary Cross Entropy with logits, used for evaluating the model's output.
loss_fn = nn.BCEWithLogitsLoss().to(device)

# Optimizer
# Initializes the Adam optimizer for the model, even though it won’t be used in evaluation.
optim = torch.optim.Adam(params=net.parameters(), lr=cfg.LEARNING_RATE)

start_epoch = 0

# Load Checkpoint File
#  Checks if there are saved checkpoints in the CKPT_DIR. If found, it loads the model and optimizer state.
if os.listdir(CKPT_DIR):
    net, optim, _ = load_net(ckpt_dir=CKPT_DIR, net=net, optim=optim)

# Evaluation Loop
# Disables gradient calculation, which is not needed during evaluation, reducing memory usage and speeding up computations.
with torch.no_grad():
    # Puts the model in evaluation mode, affecting layers like dropout and batch normalization.
    net.eval()
    # List to store loss values for each batch.
    loss_arr = list()

    for batch_idx, data in enumerate(test_loader, 1):
        # Forward Propagation
        img = data['img'].to(device)
        label = data['label'].to(device)

        # the batch of images (img) is fed through the model (output).
        output = net(img)

        # Calc Loss Function
        # The loss between the model's predictions (output) and the true labels (label) is computed.
        loss = loss_fn(output, label)
        loss_arr.append(loss.item())

        print_form = '[Test] | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
        print(print_form.format(batch_idx, test_batch_num, loss_arr[-1]))

        # Tensorboard
        # Utility functions are used to convert the tensors back to NumPy arrays, including denormalization and classification.
        img = to_numpy(denormalization(img, mean=0.5, std=0.5))
        label = to_numpy(label)
        output = to_numpy(classify_class(output))

        # The original images, labels, and model predictions are saved as .png files in the RESULTS_DIR.
        for j in range(label.shape[0]):
            crt_id = int(test_batch_num * (batch_idx - 1) + j)

            plt.imsave(os.path.join(RESULTS_DIR, f'img_{crt_id:04}.png'), img[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(RESULTS_DIR, f'label_{crt_id:04}.png'), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(RESULTS_DIR, f'output_{crt_id:04}.png'), output[j].squeeze(), cmap='gray')

# After evaluating all batches, the average loss across the entire test set is computed and printed.
print_form = '[Result] | Avg Loss: {:0.4f}'
print(print_form.format(np.mean(loss_arr)))