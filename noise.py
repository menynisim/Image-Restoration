import numpy as np


class Noise:

    def corruption(self, image):
        return self._add_gaussian_noise(image, 0, 0.2)

    ############ Private Functions ###########

    def _add_gaussian_noise(self, image, min_sigma, max_sigma):
        """
        randomly sample a value of sigma, uniformly distributed
        between min_sigma and max_sigma, followed by adding to every
        pixel of the input image a zero-mean gaussian random variable
        with standard deviation equal to sigma.
        Before returning the results, the values is rounded
        to the nearest fraction i/255 and clipped to [0, 1].
        :param image: a gray-scale image with values in the [0, 1]
        range of type float64.
        :param min_sigma: a non-negative scalar value representing
        the minimal variance of the gaussian distribution.
        :param max_sigma: a non-negative scalar value larger than
                or equal to min_sigma, representing the maximal
                variance of the gaussian distribution.
        :return: image with additive gaussian noise of unknown
                standard deviation.
        """
        sigma = np.random.uniform(min_sigma, max_sigma)
        noise = np.random.normal(0, sigma, image.shape)
        temp = np.add(image, noise)
        temp = np.round(np.multiply(temp, 255))
        result = np.divide(temp, 255)
        return np.clip(result, 0, 1)



