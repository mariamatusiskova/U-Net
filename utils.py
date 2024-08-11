#  Conversion to NumPy Array

# This specifies which functions are public and can be imported when from module import * is used.
__all__ = ['to_numpy', 'denormalization', 'classify_class']
def to_numpy(tensor):
    # detach(): Detaches the tensor from the computational graph, meaning that it won't track gradients.
    # numpy(): Converts the tensor to a NumPy array.
    # transpose(0, 2, 3, 1): Changes the tensor's shape from (Batch, Channels, Height, Width) to
    # (Batch, Height, Width, Channels) to match common image formats.
    return tensor.detach().numpy().transpose(0, 2, 3, 1)


# Reverses the normalization process applied to images during preprocessing.
def denormalization(data, mean, std):
    return (data * std) + mean


# Converts a continuous output to a binary classification
def classify_class(x):
    # x > 0.5: Compares the tensor elements to 0.5 (a common threshold for binary classification).
    # 1.0 * ...: Converts the boolean tensor to a float tensor, where True becomes 1.0 and False becomes 0.0.
    return 1.0 * (x > 0.5)
