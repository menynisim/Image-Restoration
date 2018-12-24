
class Deblur:

    def __init__(self):
        self._images_paths = None
        self._blur = None
        return

    def build(self, images_paths, blur):
        self._images_paths = images_paths
        self._blur = blur
        return

    def learn_model(self, net, num_res_blocks=5, quick_mode=False):
        """
        Train a network for deblurring an image.
        :param net: a My_Net object.
        :param num_res_blocks: Number of residual block
                               the network will consist of.
        :param quick_mode: Train model on a very little data
                          (for debugging purposes).
        :return: return a trained de-blurring model
        """
        channels = 32

        num_epochs = 10
        crop_size = (16, 16)
        return net.get_trained_model(self._images_paths, self._blur, crop_size,
                                     num_epochs, channels, num_res_blocks, quick_mode)
