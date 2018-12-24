from scipy.misc import imread as imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import os as os

GRAY_SCALE_IMAGE = 1


def read_image(filename, representation=GRAY_SCALE_IMAGE):
    """
    reads an image file and converts it into a given representation.
    :param filename: string containing the image filename to read
    :param representation: representation code,
            either 1 or 2 defining whether
             the output should be a grayscale image
             (1) or an RGB image (2).
    :return: returns an image, when the output image is
            represented by a matrix of type np.float64
            with intensities (either grayscale or RGB channel
             intensities) normalized to the range [0, 1].
    """
    im = imread(filename)
    if representation == GRAY_SCALE_IMAGE:
        return rgb2gray(im).astype(np.float64)
    return _normalize_im(im.astype(np.float64))


def save_compare_figure(images, path_to_save, i, titles=('corrupted_im', 'result_im')):
    """
    save a results im
    """
    fig = plt.figure()
    if len(images) != len(titles):
        titles = ['no title'] * len(images)

    for idx, img in enumerate(images):
        a = fig.add_subplot(1, len(images), idx + 1)
        plt.imshow(img, cmap='gray')
        a.set_title(titles[idx])

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    my_file = 'my_result_' + str(i) + '.png'
    plt.savefig(os.path.join(path_to_save, my_file))

    #plt.show()
    plt.close()


def _normalize_im(img, normal=255.0):
    """
    Normalize im to [0,1]
    :param img: the input grayscale or RGB float64 image.
    :param normal: optional - upper limit of the original range
    :return: im normalize to [0,normal]
    """
    return img / normal



