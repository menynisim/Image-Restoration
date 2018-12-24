import numpy as np

from keras.layers import Input, Convolution2D, Activation, merge
from keras.models import Model
from keras.optimizers import Adam
from image_helper import read_image


class MyNet:

    def get_trained_model(self, images_names, corrupter, crop_size, num_epochs,
                          channels, num_res_blocks=5, quick_mode=False):
        """
        Returned a trained model.
        :param images_names: a list of file paths pointing to image files.
                            We assume these paths are complete, and
                            shouldn't append anything to them.
        :param corrupter: An object with func corruption(self, image)
        :param crop_size: A tuple (height, width) specifying the
                        crop size of the patches to extract.
        :param num_epochs: The number of epochs for which the
                        optimization will run.
        :param channels: number of output channels in convolution layer.
        :param num_res_blocks: number of res blocks.
        :param quick_mode: Bool param.
        :return: Returned a trained model.
        """
        if quick_mode:
            batch_size = 10
            num_epochs = 2
            samples_per_epoch = 30
            num_valid_samples = 30

        else:
            batch_size = 100
            samples_per_epoch = 10000
            num_valid_samples = 1000

        model = self._build_nn_model(crop_size[0], crop_size[1], channels, num_res_blocks)

        self._train_model(model, images_names, corrupter, batch_size,
                          samples_per_epoch, num_epochs,
                          num_valid_samples)
        return model

    ############ Private Functions ###########

    def _load_dataset(self, file_names, batch_size, corrupter, crop_size):
        """
        outputs data_generator, a Python’s generator object which outputs random
         tuples of the form (source_batch, target_batch), where  each  output
         variable is  an  array  of  shape(batch_size, 1,height, width),
         target_batch is  made  of  clean images, and source_batch is  their
         respective randomly corrupted version according to corruption_func(im).
        :param file_names: A list of file_names of clean images
        :param batch_size: The size of the batch of images for
                each iteration of Stochastic Gradient Descent.
        :param corrupter: An object with func corruption(self, image)
        :param crop_size: A tuple (height, width) specifying the
                        crop size of the patches to extract.
        :return: data_generator, a Python’s generator object which outputs random
         tuples of the form (source_batch, target_batch), where  each  output
         variable is  an  array  of  shape(batch_size, 1,height, width),
         target_batch is  made  of  clean images, and source_batch is  their
         respective randomly corrupted version according to corruption_func(im).
        """
        cache_dic = {}
        height, width = crop_size
        source_batch = np.zeros((batch_size, 1, height, width))
        target_batch = np.zeros((batch_size, 1, height, width))

        while True:
            cur_file_names = np.random.choice(file_names, batch_size)

            for index, filename in enumerate(cur_file_names):

                if filename in cache_dic: # means its the first time we read this im
                    im = cache_dic[filename]
                else:
                    im = read_image(filename)
                    cache_dic[filename] = im

                corrupted = corrupter.corruption(im)
                crop_rows_start = np.random.random_integers(0,
                                im.shape[0] - width, 1)[0]
                crop_cols_start = np.random.random_integers(0,
                                im.shape[1] - height, 1)[0]
                target_batch[index] = im[crop_rows_start:crop_rows_start+width,
                               crop_cols_start:crop_cols_start+height]
                source_batch[index] = corrupted[crop_rows_start:crop_rows_start+width,
                               crop_cols_start:crop_cols_start+height]

            yield np.subtract(source_batch, 0.5), np.subtract(target_batch, 0.5)

    def _resblock(self, input_tensor, num_channels):
        """
        creating a residual block.
        The basic building block of ResNet is the residual block, which for
        the input X it is defined as follows: X is the input to a 3×3 convolution,
        followed by ReLU activation, and then another 3×3 convolution (this time no ReLU),
        and denote the output of the last convolution with O.
        The final output of the residual block is O + X, connecting the input to the last output,
        skipping over the middle layers.
        :param input_tensor: a symbolic input tensor.
        :param num_channels: number of output channels in convolution layer.
        :return: A symbolic output tensor of the layer configuration,
                representing a residual block.
        """
        convol = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
        relu = Activation('relu')(convol)
        convol = Convolution2D(num_channels, 3, 3, border_mode='same')(relu)
        res_block_output = merge([input_tensor, convol], mode='sum')
        return res_block_output

    def _build_nn_model(self, height, width, num_channels, num_res_blocks):
        """
        Returns the complete neural network model - untrained Keras model,
        with input dimension the shape of (1, height, width),
        and all convolutional layers (including residual
        blocks) with number of output channels equal to num_channels,
        except the very last convolutional
        layer which should have a single output channel.
        :param height: the height parameter in the input tensor.
        :param width: the width parameter in the input tensor.
        :param num_channels: number of output channels in convolution layer.
        :param num_res_blocks: number of res blocks.
        :return: Returns the complete neural network model.
        """
        my_input = Input(shape=(1, height, width))
        convol = Convolution2D(num_channels, 3, 3, border_mode='same')(my_input)
        relu_start = Activation('relu')(convol)

        block = relu_start
        for _ in range(num_res_blocks):
            block = self._resblock(block, num_channels)

        loop_result = merge([relu_start, block], mode='sum')
        my_output = Convolution2D(1, 3, 3, border_mode='same')(loop_result)
        model = Model(input=my_input, output=my_output)

        return model

    def _train_model(self, model, images, corrupter, batch_size,
                     samples_per_epoch, num_epochs, num_valid_samples):
        """
        :param model: a general neural network model for image restoration.
        :param images:  a list of file paths pointing to image files.
                        We assume these paths are complete, and
                        should append anything to them.
        :param corrupter: An object with func corruption(self, image)
        :param batch_size: the size of the batch of examples for each iteration of SGD.
        :param samples_per_epoch: The number of samples in each epoch
                                (actual samples, not batches!)
        :param num_epochs: The number of epochs for which the optimization will run.
        :param num_valid_samples: The number of samples in the validation set
                                to test on after every epoch
        """
        shuffled_im = np.random.permutation(images)
        ims_for_train = shuffled_im[:int(len(shuffled_im) * 0.8)]
        ims_for_valid = shuffled_im[int(len(shuffled_im) * 0.8):]

        crop_size = model.input_shape[2:]
        model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))

        train_set = self._load_dataset(ims_for_train, batch_size, corrupter, crop_size)
        valid_set = self._load_dataset(ims_for_valid, batch_size, corrupter, crop_size)
        model.fit_generator(train_set,samples_per_epoch=samples_per_epoch,
                            nb_epoch=num_epochs, validation_data=valid_set,
                            nb_val_samples=num_valid_samples)

