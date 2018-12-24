import numpy as np
from keras.layers import Input
from keras.models import Model


class Restore:

    def restore_image(self, corrupted_image, base_model):
        """
        Restore full corrupted image of any size.
        :param corrupted_image: a grayscale image of shape (height, width)
        and with values in the [0, 1] range of type float64.
        :param base_model: a neural network trained to restore small patches.
        The input and output of the network are images with values
        in the [−0.5, 0.5] range.
        :return: An uncorrupted image.
        """
        height, width = corrupted_image.shape
        a = Input(shape=(1, height, width))
        b = base_model(a)
        new_model = Model(input=a, output=b)

        corrupted_image = corrupted_image.reshape((1, height, width))
        corrupted_image = np.subtract(corrupted_image, 0.5)
        img = new_model.predict(corrupted_image[np.newaxis,
                                                ...])[0].reshape((height, width))
        img = np.add(img, 0.5)
        return np.clip(img, 0, 1).astype(np.float64)

