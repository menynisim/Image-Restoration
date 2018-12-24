import os
import random


def list_paths(path, use_shuffle=True):
    """
    Returns a list of paths to images found at the specified directory.
    :param path: path to a directory to search for images.
    :param use_shuffle: option to shuffle order of files. Uses a fixed shuffled order.
    :return: a list of paths to images found at the specified directory.
    """
    real_path = _find_real_path(path)

    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']

    images = list(map(lambda x: os.path.join(real_path, x), filter(is_image, os.listdir(real_path))))

    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def _find_real_path(path):
    """
    Returns the relative path to the script's location.
    :param path: a string representation of a path.
    :return: the relative path to the script's location.
    """
    return os.path.join(os.path.dirname(__file__), path)
