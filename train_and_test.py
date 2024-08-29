import os
import numpy as np
from sympy.physics.control.control_plots import plt
from torch import no_grad
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from config import Config
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from model import UNet
from dataset_isbi import ISBIDataset
from transformation.grayscale_normalization import GrayscaleNormalization
from transformation.random_flip import RandomFlip, RandomRotate
from transformation.to_tensor import ToTensor
from utils import *


config = Config()

train_transform = transforms.Compose([
    GrayscaleNormalization(),
    RandomFlip(),
    # RandomRotate(),
    ToTensor(),
])

val_transform = transforms.Compose([
    GrayscaleNormalization(),
    ToTensor(),
])

# Composes a series of transformations
# GrayscaleNormalization: Normalizes the grayscale image.
# ToTensor: Converts the image and label data from NumPy arrays to PyTorch tensors.
test_transform = transforms.Compose([
    GrayscaleNormalization(),
    ToTensor(),
])


train_dataset = ISBIDataset('./data/train_img_data/', './data/train_label_data/', train_transform)
val_dataset = ISBIDataset('./data/val_img_data/', './data/val_label_data/', val_transform)
test_dataset = ISBIDataset('./data/test_img_data/', './data/test_label_data/', test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

train_data_num = len(train_dataset)
val_data_num = len(val_dataset)
# Total number of samples in the test dataset.
test_data_num = len(test_dataset)

train_batch_num = int(np.ceil(train_data_num / config.BATCH_SIZE))
val_batch_num = int(np.ceil(val_data_num / config.BATCH_SIZE))
# The total number of batches for testing, calculated by dividing the number of test samples by the batch size.
test_batch_num = int(np.ceil(test_data_num / config.BATCH_SIZE))

model = UNet()

# binary classification tasks
# sigmoid activation
# binary cross-entropy loss
loss_fun = nn.BCEWithLogitsLoss()

# Adaptive Moment Estimation
optimizer = Adam(params=model.parameters(), lr=config.LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# for tracking and debugging during the training phase
train_writer = SummaryWriter(log_dir='./data/train_log_data/')
val_writer = SummaryWriter(log_dir='./data/val_log_data/')

start_epoch = 0
num_epochs = config.NUM_EPOCHS
#

# Each epoch runs through the entire dataset using mini-batches.
for epoch in range(start_epoch + 1, num_epochs + 1):
    # set the model to training mode
    model.train()
    train_loss = list()

    # train loop
    for batch_idx, data in enumerate(train_dataloader, 1):
        image, label = data['image'], data['label']
        # print(f"Image shape: {image.shape}")
        # print(f"Label shape: {label.shape}")

        # generate predictions
        predictions = model(image)

        # calculate loss
        loss = loss_fun(predictions, label)

        # reset gradients
        optimizer.zero_grad()

        # update weights and biases, compute gradients
        loss.backward()
        optimizer.step()

        # save data and print results
        train_loss.append(loss.item())
        print_form = '[Train] | Epoch: {:0>2d} / {:0>2d} | Batch: {:0>2d} / {:0>2d} | Loss: {:.4f}'
        print(print_form.format(epoch, num_epochs, batch_idx, train_batch_num, train_loss[-1]))

        # tensorboard
        image = to_numpy(denormalization(image, mean=0.5, std=0.5))
        label = to_numpy(label)
        predictions = to_numpy(classify_class(predictions))

        global_step = train_batch_num * (epoch - 1) + batch_idx
        train_writer.add_image(tag='image', img_tensor=image, global_step=global_step, dataformats='NHWC')
        train_writer.add_image(tag='label', img_tensor=label, global_step=global_step, dataformats='NHWC')
        train_writer.add_image(tag='predictions', img_tensor=predictions, global_step=global_step, dataformats='NHWC')

    train_loss_avg = np.mean(train_loss)
    train_writer.add_scalar(tag='loss', scalar_value=train_loss_avg, global_step=epoch)


    # validation loop
    with no_grad():
        model.eval()
        val_loss = list()

        for batch_idx, data in enumerate(val_dataloader, 1):
            image, label = data['image'], data['label']

            # generate predictions
            predictions = model(image)

            # calculate loss
            loss = loss_fun(predictions, label)

            # save data and print results
            val_loss.append(loss.item())
            print_form = '[Validation] | Epoch: {:0>2d} / {:0>2d} | Batch: {:0>2d} / {:0>2d} | Loss: {:.4f}'
            print(print_form.format(epoch, num_epochs, batch_idx, val_batch_num, val_loss[-1]))

            # tensorboard
            image = to_numpy(denormalization(image, mean=0.5, std=0.5))
            label = to_numpy(label)
            predictions = to_numpy(classify_class(predictions))

            global_step = val_batch_num * (epoch - 1) + batch_idx
            val_writer.add_image(tag='image', img_tensor=image, global_step=global_step, dataformats='NHWC')
            val_writer.add_image(tag='label', img_tensor=label, global_step=global_step, dataformats='NHWC')
            val_writer.add_image(tag='predictions', img_tensor=predictions, global_step=global_step, dataformats='NHWC')

    val_loss_avg = np.mean(val_loss)
    val_writer.add_scalar(tag='loss', scalar_value=val_loss_avg, global_step=epoch)

    print_form = '[Epoch {:0>2d}] Training Avg Loss: {:.4f} | Validation Avg Loss: {:.4f}'
    print(print_form.format(epoch, train_loss_avg, val_loss_avg))

    scheduler.step()

train_writer.close()
val_writer.close()

# evaluation loop
start_epoch = 0
with no_grad():
    model.eval()
    test_loss = list()

    for batch_idx, data in enumerate(test_dataloader, 1):
        image, label = data['image'], data['label']

        # generate predictions
        predictions = model(image)

        # calculate loss
        loss = loss_fun(predictions, label)

        # save data and print results
        test_loss.append(loss.item())
        print_form = '[Test] | Batch: {:0>2d} / {:0>2d} | Loss: {:.4f}'
        print(print_form.format(batch_idx, test_batch_num, test_loss[-1]))

        # tensorboard
        # Utility functions are used to convert the tensors back to NumPy arrays, including denormalization and classification.
        image = to_numpy(denormalization(image, mean=0.5, std=0.5))
        label = to_numpy(label)
        predictions = to_numpy(classify_class(predictions))

        # The original images, labels, and model predictions are saved as .png files in the RESULTS_DIR.
        for j in range(label.shape[0]):
            crt_id = int(test_batch_num * (batch_idx - 1) + j)

            plt.imsave(os.path.join('./data/eval_results/', f'image_{crt_id:04}.png'), image[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join('./data/eval_results/', f'label_{crt_id:04}.png'), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join('./data/eval_results/', f'predictions_{crt_id:04}.png'), predictions[j].squeeze(), cmap='gray')

# After evaluating all batches, the average loss across the entire test set is computed and printed.
print_form = '[Result] | Avg Loss: {:0.4f}'
print(print_form.format(np.mean(test_loss)))