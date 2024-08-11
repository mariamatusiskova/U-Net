class GrayscaleNormalization:
    def __init__(self, mean=0.5, std=0.5):
        # mean value used for normalizing data
        self.mean = mean
        # the standard deviation used for normalization
        self.std = std

    def __call__(self, data: dict) -> dict:
        image, label = data['image'], data['label']

        # normalize an image by subtracting the mean and dividing by the standard deviation
        image = (image - self.mean) / self.std

        return_dict = {
            'image': image,
            'label': label,
        }

        return return_dict
