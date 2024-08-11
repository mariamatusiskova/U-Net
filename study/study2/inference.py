# Used for measuring the execution time of small code snippets.
from timeit import default_timer as timer
import numpy as np
# A GPU-accelerated drop-in replacement for NumPy.
import cupy as cp
import torch
# Wraps tensors and records operations for automatic differentiation.
from torch.autograd import Variable
# A utility function from utils.py for grouping elements.
from utils import groupby

# This function prepares image patches centered around specific positions for further processing.
# img: The image to process (assumed to be 2048x2048 pixels).
# pos: A list of positions (x, y) where the patches are extracted.
# half_size: Half the size of the square patch to be extracted around each position (default 15, so the full patch is 30x30).
# device: The GPU device ID to use.
def preprocess(img, pos, half_size=15, device=0):
	with cp.cuda.Device(device):
        # pos_x_left, pos_x_right: Left and right x-coordinates of the patches.
		pos_x_left, pos_x_right = pos[:,0]-half_size, pos[:,0]+half_size
        # pos_y_left, pos_y_right: Top and bottom y-coordinates.
		pos_y_left, pos_y_right = pos[:,1]-half_size, pos[:,1]+half_size

        # A boolean mask that checks if patches fit within image boundaries.
		SELECT_MAP = (pos_x_left>=0)*(pos_y_left>=0)*(pos_x_right<2048)*(pos_y_right<2048)
        # Indices of valid positions.
		SELECT_INDEX = cp.where(SELECT_MAP>0)[0]

        # The valid positions are then used to update
		pos_x_left, pos_x_right, pos_y_left, pos_y_right = pos_x_left[SELECT_INDEX], pos_x_right[SELECT_INDEX], pos_y_left[SELECT_INDEX], pos_y_right[SELECT_INDEX]

		pos = pos[SELECT_INDEX]

        # Expanding Dimensions
        # Expands dimensions of pos_x and pos_y to prepare for indexing.
		# shape should be dx * dy * dz
		pos_x = pos_x_left
		pos_x = cp.expand_dims(pos_x, axis=0)
		adding = cp.expand_dims(cp.arange(2*half_size), 1)
		pos_x = pos_x+adding

		pos_y = pos_y_left
		pos_y = cp.expand_dims(pos_y, axis=0)
		adding = cp.expand_dims(cp.arange(2*half_size), 1)
		pos_y = pos_y+adding

        # Extracting Patches
        # Extracts image patches from the original image using the calculated positions.
		# x * y * N
		_x = img[pos_x[:, cp.newaxis, :], pos_y]

# Extracted patches (_x) and the positions (pos).
	return _x.get(), pos.get() #, groups_start, groups_end


# This function calculates the interface (bounding coordinates) of the patches for use later in the binary map.
# pos_x, pos_y: The center positions of the patches.
# half_size: Half the size of the patch.
def patch_interface(pos_x, pos_y, half_size):
    # Calculate Bounding Box:
    # Computes the left and right (or top and bottom) bounds of the patch for each position.
	pos_x_left = pos_x - half_size
	pos_x_left = np.expand_dims(pos_x_left, axis=0)
	adding = np.expand_dims(np.arange(2*half_size), 1)
	pos_x = pos_x_left + adding

	pos_y_left = pos_y - half_size
	pos_y_left = np.expand_dims(pos_y_left, axis=0)
	adding = np.expand_dims(np.arange(2*half_size), 1)
	pos_y = pos_y_left+adding

    # The patch bounds (pos_x, pos_y) for both x and y dimensions.
	# x * y * N
	return pos_x[:, np.newaxis, :], pos_y


from model_convention import unet

# This function runs the deep learning model (U-Net) on the preprocessed image patches to generate a binary output.
# img: The input image.
# state_path: Path to the model's saved state (weights).
# device: The GPU device ID to use.
def inference(img, state_path, device=0):
    # Loads the U-Net model and its pre-trained weights.
	model = unet().cuda(device)
	state = torch.load(state_path)
	model.load_state_dict(state)

    # Initializes an empty binary map of the same shape as the input image.
	binary = np.zeros(img.shape)

    # Generates a grid of positions (strided by 60 pixels).
	stride, patch_size = 60, 60

	with cp.cuda.Device(device):
		img = cp.asarray(img.astype(np.uint16))

		pos_x = np.arange(0, 2048, stride)
		pos_y = np.arange(0, 2048, stride)
		vx, vy = np.meshgrid(pos_x, pos_y)
		pos = cp.asarray(np.stack([vx, vy]).reshape((2, -1)).transpose([1,0]))


		X, pos = preprocess(img, pos, half_size=patch_size//2, device=device)

		X = X.transpose([2, 0, 1])

		indices = np.arange(len(X)).tolist()
        # Uses the groupby function to batch process the patches.
		groups = groupby(indices, 100, key='mini')


		out_list = []
        # Converts them to PyTorch tensors and runs them through the model.
		for index, group in enumerate(groups):
			out = X[group]
			_pos = pos[group]

            # The model's output is post-processed and added to the binary map.
			out = Variable(torch.from_numpy(np.expand_dims(out, 1).astype(np.float32)).cuda(device))
			R = model(out)
			R = R.max(1)[1]
			patch = R.cpu().numpy().transpose([1,2,0])
			binary[patch_interface(_pos[:,0], _pos[:,1], patch_size//2)] = patch

    # Converts the binary map to a uint8 format (0-255) for further use (e.g., saving as an image).
	binary = (binary*255).astype('uint8')
    # The final binary map (binary).
	return binary