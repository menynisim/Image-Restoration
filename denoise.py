
class Denoise:

    def __init__(self):
        self._images_paths = None
        self._noise = None
        return

    def build(self, images_paths, noise):
        self._images_paths = images_paths
        self._noise = noise
        return

    def learn_model(self, net, num_res_blocks=5, quick_mode=False):
        """
        Train a network for de-noising an image.
        :param net: a My_Net object.
        :param num_res_blocks: Number of residual block
                               the network will consist of.
        :param quick_mode: Train model on a very little data
                          (for debugging purposes).
        :return: return a trained de-noising model
        """

        num_epochs = 5
        channels = 48

        crop_size = (24, 24)
        return net.get_trained_model(self._images_paths, self._noise,
                                     crop_size, num_epochs, channels, num_res_blocks, quick_mode)

