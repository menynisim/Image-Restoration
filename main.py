import os
import read_paths
from my_net import MyNet
from restore import Restore
from image_helper import read_image, save_compare_figure
from keras.models import load_model
from nets_factory import select_model
from get_input import get_input

TO_REPLACE = '#'

TEST_PATH = TO_REPLACE + '_dataset\\test'
TRAIN_PATH = TO_REPLACE + '_dataset\\train'
RESULTS_PATH = TO_REPLACE + '_dataset\\results'
NAME_TO_SAVE = TO_REPLACE + '_model.h5'

FULL_MODE = 'full'
QUICK_MODE = 'quick'
MY_MODES = [QUICK_MODE, FULL_MODE]


def select_mode():
    mode_options = list('to ' + mode_option + ' press ' + str(i) for i, mode_option in enumerate(MY_MODES))
    message = 'Please select the requested mode:\n'
    mode = get_input(message, mode_options, MY_MODES)
    return mode == QUICK_MODE


def create_model(my_model, corrupter, net_name, name_to_save):
    net = MyNet()

    paths = read_paths.list_paths(TRAIN_PATH.replace(TO_REPLACE, net_name))
    my_model.build(paths, corrupter)

    mode = select_mode()
    trained_model = my_model.learn_model(net, quick_mode=mode)

    trained_model.save(name_to_save)
    return trained_model


def restore_image(trained_model, restore, corrupter, im, i, net_name):
    im_corrupted = corrupter.corruption(im)
    im_restored = restore.restore_image(im_corrupted, trained_model)
    save_compare_figure([im_corrupted, im_restored], RESULTS_PATH.replace(TO_REPLACE, net_name), i)


def restore_images(trained_model, corrupter, net_name):
    paths = read_paths.list_paths(TEST_PATH.replace(TO_REPLACE, net_name))
    restore = Restore()
    for i, path in enumerate(paths):
        restore_image(trained_model, restore, corrupter, read_image(path), i, net_name)


def main():

    my_model, corrupter, net_name = select_model()

    name_to_save = NAME_TO_SAVE.replace(TO_REPLACE, net_name)
    if os.path.isfile(name_to_save):
        trained_model = load_model(name_to_save)
    else:
        trained_model = create_model(my_model, corrupter, net_name, name_to_save)

    restore_images(trained_model, corrupter, net_name)


if __name__ == "__main__":
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
