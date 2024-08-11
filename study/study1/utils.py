import os
import torch
from config import CKPT_DIR

# This specifies which functions are public and can be imported when from module import * is used.
__all__ = ['to_numpy', 'denormalization', 'classify_class', 'save_net', 'load_net']

#  Conversion to NumPy Array
def to_numpy(tensor):
    # detach(): Detaches the tensor from the computational graph, meaning that it won't track gradients.
    # numpy(): Converts the tensor to a NumPy array.
    # transpose(0, 2, 3, 1): Changes the tensor's shape from (Batch, Channels, Height, Width) to
    # (Batch, Height, Width, Channels) to match common image formats.
    return tensor.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # (Batch, H, W, C)

# Reverses the normalization process applied to images during preprocessing.
def denormalization(data, mean, std):
    return (data * std) + mean

# Converts a continuous output to a binary classification
def classify_class(x):
    # x > 0.5: Compares the tensor elements to 0.5 (a common threshold for binary classification).
    # 1.0 * ...: Converts the boolean tensor to a float tensor, where True becomes 1.0 and False becomes 0.0.
    return 1.0 * (x > 0.5)

# Saving a Model Checkpoint
# Saves the current state of the model and optimizer to a checkpoint file.
def save_net(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # net.state_dict(): Retrieves the model's parameters.
    # optim.state_dict(): Retrieves the optimizer's state.
    # torch.save: Saves the model and optimizer states to a .pth file named according to the epoch.
    torch.save(
        {'net': net.state_dict(), 'optim': optim.state_dict()},
        os.path.join(ckpt_dir, f'model_epoch{epoch:04}.pth'),
    )

# Loading a Model Checkpoint
# Loads the model and optimizer states from the most recent checkpoint.
def load_net(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_list = os.listdir(ckpt_dir)
    # Sorts the list of checkpoints by epoch number, ensuring the latest one is selected.
    ckpt_list.sort(key=lambda fname: int(''.join(filter(str.isdigit, fname))))

    ckpt_path = os.path.join(CKPT_DIR, ckpt_list[-1])
    # Loads the model and optimizer states from the checkpoint file.
    model_dict = torch.load(ckpt_path)
    print(f'* Load {ckpt_path}')

    # Loads the model parameters.
    net.load_state_dict(model_dict['net'])
    # Loads the optimizer's state.
    optim.load_state_dict(model_dict['optim'])
    epoch = int(''.join(filter(str.isdigit, ckpt_list[-1])))

    # The model, optimizer, and the epoch number from which to resume training.
    return net, optim, epoch