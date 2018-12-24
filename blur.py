import numpy as np
from scipy.ndimage.filters import convolve
from skimage.draw import line


class Blur:

    def corruption(self, image):
        return self._random_motion_blur(image, [3, 5, 7, 9])

    ############ Private Functions ###########

    def _random_motion_blur(self, image, list_of_kernel_sizes):
        """
        samples an angle at uniform from the range [0, π),
        chooses a kernel size at uniform from the list list_of_kernel_sizes,
        and return add_motion_blur(image, kernel_size, angle).
        :param image: a gray-scale image with values in the
        [0, 1] range of type float64.
        :param list_of_kernel_sizes: a list of odd integers.
        :return: return add_motion_blur(image, kernel_size, angle).
        """
        angle = np.random.uniform(0, np.math.pi)
        kernel_size = np.random.choice(list_of_kernel_sizes)
        return self._add_motion_blur(image, kernel_size, angle)

    ############ Private Functions ###########

    def _add_motion_blur(self, image, kernel_size, angle):
        """
        simulate motion blur on the given image using a square kernel of
        size kernel_size where the line has the given angle in radians.
        :param image: a gray-scale image with values in the
        [0, 1] range of type float64.
        :param kernel_size: an odd integer specifying the size of
        the kernel (even integers are ill-defined).
        :param angle: an angle in radians in the range [0, π).
        :return: im after motion blur on it using a square kernel of
        size kernel_size where the line has the
        given angle in radians.
        """
        kernel = self._motion_blur_kernel(kernel_size, angle)
        return convolve(image, kernel)

    # Note: I didn't write this function
    def _motion_blur_kernel(self, kernel_size, angle):
        """
        Returns a 2D image kernel for motion blur effect.
        Arguments:
        kernel_size -- the height and width of the kernel. Controls strength of blur.
        angle -- angle in the range [0, np.pi) for the direction of the motion.
        """
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be an odd number!')
        if angle < 0 or angle > np.pi:
            raise ValueError('angle must be between 0 (including) and pi (not including)')
        norm_angle = 2.0 * angle / np.pi
        if norm_angle > 1:
            norm_angle = 1 - norm_angle
        half_size = kernel_size // 2
        if abs(norm_angle) == 1:
            p1 = (half_size, 0)
            p2 = (half_size, kernel_size - 1)
        else:
            alpha = np.tan(np.pi * 0.5 * norm_angle)
            if abs(norm_angle) <= 0.5:
                p1 = (2 * half_size, half_size - int(round(alpha * half_size)))
                p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
            else:
                alpha = np.tan(np.pi * 0.5 * (1 - norm_angle))
                p1 = (half_size - int(round(alpha * half_size)), 2 * half_size)
                p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
        rr, cc = line(p1[0], p1[1], p2[0], p2[1])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
        kernel[rr, cc] = 1.0
        kernel /= kernel.sum()
        return kernel
