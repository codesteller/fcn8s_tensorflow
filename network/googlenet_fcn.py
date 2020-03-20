import tensorflow as tf
import numpy as np

import network.nlayers as nlayers
from network.nlayers import networkConfig


class Inceptionv1FCN8s:
    def __init__(self, config):
        self.in_data = tf.placeholder(tf.float32, name='in_data', shape=[config.batch_size,
                                                                         config.image_width,
                                                                         config.image_height,
                                                                         config.image_depth], )
        self.labels = tf.placeholder(tf.float32, name='labels', shape=[config.batch_size,
                                                                       config.image_width,
                                                                       config.image_height,
                                                                       config.n_classes])
        self.logits = None
        self.config = config

    def core_net(self):
        conv1_7x7_s2 = nlayers.conv2d('conv1_7x7_s2', self.in_data, 64, 7, 2)
        pool1_3x3_s2 = nlayers.max_pool('pool1_3x3_s2', conv1_7x7_s2, 3, 2)
        pool1_norm1 = nlayers.lrn('pool1_norm1', pool1_3x3_s2)
        conv2_3x3_reduce = nlayers.conv2d(
            'conv2_3x3_reduce', pool1_norm1, 64, 1, 1)
        conv2_3x3 = nlayers.conv2d('conv2_3x3', conv2_3x3_reduce, 192, 3, 1)
        conv2_norm2 = nlayers.lrn('conv2_norm2', conv2_3x3)
        pool2_3x3_s2 = nlayers.max_pool('pool2_3x3_s2', conv2_norm2, 3, 2)

        inception_3a_1x1 = nlayers.conv2d(
            'inception_3a_1x1', pool2_3x3_s2, 64, 1, 1)
        inception_3a_3x3_reduce = nlayers.conv2d(
            'inception_3a_3x3_reduce', pool2_3x3_s2, 96, 1, 1)
        inception_3a_3x3 = nlayers.conv2d(
            'inception_3a_3x3', inception_3a_3x3_reduce, 128, 3, 1)
        inception_3a_5x5_reduce = nlayers.conv2d(
            'inception_3a_5x5_reduce', pool2_3x3_s2, 16, 1, 1)
        inception_3a_5x5 = nlayers.conv2d(
            'inception_3a_5x5', inception_3a_5x5_reduce, 32, 5, 1)
        inception_3a_pool = nlayers.max_pool(
            'inception_3a_pool', pool2_3x3_s2, 3, 1)
        inception_3a_pool_proj = nlayers.conv2d(
            'inception_3a_pool_proj', inception_3a_pool, 32, 1, 1)
        inception_3a_output = nlayers.concat('inception_3a_output',
                                             [inception_3a_1x1, inception_3a_3x3, inception_3a_5x5,
                                              inception_3a_pool_proj])

        inception_3b_1x1 = nlayers.conv2d(
            'inception_3b_1x1', inception_3a_output, 128, 1, 1)
        inception_3b_3x3_reduce = nlayers.conv2d(
            'inception_3b_3x3_reduce', inception_3a_output, 128, 1, 1)
        inception_3b_3x3 = nlayers.conv2d(
            'inception_3b_3x3', inception_3b_3x3_reduce, 192, 3, 1)
        inception_3b_5x5_reduce = nlayers.conv2d(
            'inception_3b_5x5_reduce', inception_3a_output, 32, 1, 1)
        inception_3b_5x5 = nlayers.conv2d(
            'inception_3b_5x5', inception_3b_5x5_reduce, 96, 5, 1)
        inception_3b_pool = nlayers.max_pool(
            'inception_3b_pool', inception_3a_output, 3, 1)
        inception_3b_pool_proj = nlayers.conv2d(
            'inception_3b_pool_proj', inception_3b_pool, 64, 1, 1)
        inception_3b_output = nlayers.concat('inception_3b_output',
                                             [inception_3b_1x1, inception_3b_3x3, inception_3b_5x5,
                                              inception_3b_pool_proj])

        pool3_3x3_s2 = nlayers.max_pool(
            'pool3_3x3_s2', inception_3b_output, 3, 2)
        inception_4a_1x1 = nlayers.conv2d(
            'inception_4a_1x1', pool3_3x3_s2, 192, 1, 1)
        inception_4a_3x3_reduce = nlayers.conv2d(
            'inception_4a_3x3_reduce', pool3_3x3_s2, 96, 1, 1)
        inception_4a_3x3 = nlayers.conv2d(
            'inception_4a_3x3', inception_4a_3x3_reduce, 208, 3, 1)
        inception_4a_5x5_reduce = nlayers.conv2d(
            'inception_4a_5x5_reduce', pool3_3x3_s2, 16, 1, 1)
        inception_4a_5x5 = nlayers.conv2d(
            'inception_4a_5x5', inception_4a_5x5_reduce, 48, 5, 1)
        inception_4a_pool = nlayers.max_pool(
            'inception_4a_pool', pool3_3x3_s2, 3, 1)
        inception_4a_pool_proj = nlayers.conv2d(
            'inception_4a_pool_proj', inception_4a_pool, 64, 1, 1)
        inception_4a_output = nlayers.concat('inception_4a_output',
                                             [inception_4a_1x1, inception_4a_3x3, inception_4a_5x5,
                                              inception_4a_pool_proj])

        inception_4b_1x1 = nlayers.conv2d(
            'inception_4b_1x1', inception_4a_output, 160, 1, 1)
        inception_4b_3x3_reduce = nlayers.conv2d(
            'inception_4b_3x3_reduce', inception_4a_output, 112, 1, 1)
        inception_4b_3x3 = nlayers.conv2d(
            'inception_4b_3x3', inception_4b_3x3_reduce, 224, 3, 1)
        inception_4b_5x5_reduce = nlayers.conv2d(
            'inception_4b_5x5_reduce', inception_4a_output, 24, 1, 1)
        inception_4b_5x5 = nlayers.conv2d(
            'inception_4b_5x5', inception_4b_5x5_reduce, 64, 5, 1)
        inception_4b_pool = nlayers.max_pool(
            'inception_4b_pool', inception_4a_output, 3, 1)
        inception_4b_pool_proj = nlayers.conv2d(
            'inception_4b_pool_proj', inception_4b_pool, 64, 1, 1)
        inception_4b_output = nlayers.concat('inception_4b_output',
                                             [inception_4b_1x1, inception_4b_3x3, inception_4b_5x5,
                                              inception_4b_pool_proj])

        inception_4c_1x1 = nlayers.conv2d(
            'inception_4c_1x1', inception_4b_output, 128, 1, 1)
        inception_4c_3x3_reduce = nlayers.conv2d(
            'inception_4c_3x3_reduce', inception_4b_output, 128, 1, 1)
        inception_4c_3x3 = nlayers.conv2d(
            'inception_4c_3x3', inception_4c_3x3_reduce, 256, 3, 1)
        inception_4c_5x5_reduce = nlayers.conv2d(
            'inception_4c_5x5_reduce', inception_4b_output, 24, 1, 1)
        inception_4c_5x5 = nlayers.conv2d(
            'inception_4c_5x5', inception_4c_5x5_reduce, 64, 5, 1)
        inception_4c_pool = nlayers.max_pool(
            'inception_4c_pool', inception_4b_output, 3, 1)
        inception_4c_pool_proj = nlayers.conv2d(
            'inception_4c_pool_proj', inception_4c_pool, 64, 1, 1)
        inception_4c_output = nlayers.concat('inception_4c_output',
                                             [inception_4c_1x1, inception_4c_3x3, inception_4c_5x5,
                                              inception_4c_pool_proj])

        inception_4d_1x1 = nlayers.conv2d(
            'inception_4d_1x1', inception_4c_output, 112, 1, 1)
        inception_4d_3x3_reduce = nlayers.conv2d(
            'inception_4d_3x3_reduce', inception_4c_output, 144, 1, 1)
        inception_4d_3x3 = nlayers.conv2d(
            'inception_4d_3x3', inception_4d_3x3_reduce, 288, 3, 1)
        inception_4d_5x5_reduce = nlayers.conv2d(
            'inception_4d_5x5_reduce', inception_4c_output, 32, 1, 1)
        inception_4d_5x5 = nlayers.conv2d(
            'inception_4d_5x5', inception_4d_5x5_reduce, 64, 5, 1)
        inception_4d_pool = nlayers.max_pool(
            'inception_4d_pool', inception_4c_output, 3, 1)
        inception_4d_pool_proj = nlayers.conv2d(
            'inception_4d_pool_proj', inception_4d_pool, 64, 1, 1)
        inception_4d_output = nlayers.concat('inception_4d_output',
                                             [inception_4d_1x1, inception_4d_3x3, inception_4d_5x5,
                                              inception_4d_pool_proj])

        inception_4e_1x1 = nlayers.conv2d(
            'inception_4e_1x1', inception_4d_output, 256, 1, 1)
        inception_4e_3x3_reduce = nlayers.conv2d(
            'inception_4e_3x3_reduce', inception_4d_output, 160, 1, 1)
        inception_4e_3x3 = nlayers.conv2d(
            'inception_4e_3x3', inception_4e_3x3_reduce, 320, 3, 1)
        inception_4e_5x5_reduce = nlayers.conv2d(
            'inception_4e_5x5_reduce', inception_4d_output, 32, 1, 1)
        inception_4e_5x5 = nlayers.conv2d(
            'inception_4e_5x5', inception_4e_5x5_reduce, 128, 5, 1)
        inception_4e_pool = nlayers.max_pool(
            'inception_4e_pool', inception_4d_output, 3, 1)
        inception_4e_pool_proj = nlayers.conv2d(
            'inception_4e_pool_proj', inception_4e_pool, 128, 1, 1)
        inception_4e_output = nlayers.concat('inception_4e_output',
                                             [inception_4e_1x1, inception_4e_3x3, inception_4e_5x5,
                                              inception_4e_pool_proj])

        pool4_3x3_s2 = nlayers.max_pool(
            'pool4_3x3_s2', inception_4e_output, 3, 2)
        inception_5a_1x1 = nlayers.conv2d(
            'inception_5a_1x1', pool4_3x3_s2, 256, 1, 1)
        inception_5a_3x3_reduce = nlayers.conv2d(
            'inception_5a_3x3_reduce', pool4_3x3_s2, 160, 1, 1)
        inception_5a_3x3 = nlayers.conv2d(
            'inception_5a_3x3', inception_5a_3x3_reduce, 320, 3, 1)
        inception_5a_5x5_reduce = nlayers.conv2d(
            'inception_5a_5x5_reduce', pool4_3x3_s2, 32, 1, 1)
        inception_5a_5x5 = nlayers.conv2d(
            'inception_5a_5x5', inception_5a_5x5_reduce, 128, 5, 1)
        inception_5a_pool = nlayers.max_pool(
            'inception_5a_pool', pool4_3x3_s2, 3, 1)
        inception_5a_pool_proj = nlayers.conv2d(
            'inception_5a_pool_proj', inception_5a_pool, 128, 1, 1)
        inception_5a_output = nlayers.concat('inception_5a_output',
                                             [inception_5a_1x1, inception_5a_3x3, inception_5a_5x5,
                                              inception_5a_pool_proj])

        inception_5b_1x1 = nlayers.conv2d(
            'inception_5b_1x1', inception_5a_output, 384, 1, 1)
        inception_5b_3x3_reduce = nlayers.conv2d(
            'inception_5b_3x3_reduce', inception_5a_output, 192, 1, 1)
        inception_5b_3x3 = nlayers.conv2d(
            'inception_5b_3x3', inception_5b_3x3_reduce, 384, 3, 1)
        inception_5b_5x5_reduce = nlayers.conv2d(
            'inception_5b_5x5_reduce', inception_5a_output, 48, 1, 1)
        inception_5b_5x5 = nlayers.conv2d(
            'inception_5b_5x5', inception_5b_5x5_reduce, 128, 5, 1)
        inception_5b_pool = nlayers.max_pool(
            'inception_5b_pool', inception_5a_output, 3, 1)
        inception_5b_pool_proj = nlayers.conv2d(
            'inception_5b_pool_proj', inception_5b_pool, 128, 1, 1)
        inception_5b_output = nlayers.concat('inception_5b_output',
                                             [inception_5b_1x1, inception_5b_3x3, inception_5b_5x5,
                                              inception_5b_pool_proj])

        pool5_3x3_s2 = nlayers.max_pool(
            'pool5_3x3_s2', inception_5b_output, 3, 2)
        inception_6a_1x1 = nlayers.conv2d(
            'inception_6a_1x1', pool5_3x3_s2, 256, 1, 1)
        inception_6a_3x3_reduce = nlayers.conv2d(
            'inception_6a_3x3_reduce', pool5_3x3_s2, 160, 1, 1)
        inception_6a_3x3 = nlayers.conv2d(
            'inception_6a_3x3', inception_6a_3x3_reduce, 320, 3, 1)
        inception_6a_5x5_reduce = nlayers.conv2d(
            'inception_6a_5x5_reduce', pool5_3x3_s2, 32, 1, 1)
        inception_6a_5x5 = nlayers.conv2d(
            'inception_6a_5x5', inception_6a_5x5_reduce, 128, 5, 1)
        inception_6a_pool = nlayers.max_pool(
            'inception_6a_pool', pool5_3x3_s2, 3, 1)
        inception_6a_pool_proj = nlayers.conv2d(
            'inception_6a_pool_proj', inception_6a_pool, 128, 1, 1)
        inception_6a_output = nlayers.concat('inception_6a_output',
                                             [inception_6a_1x1, inception_6a_3x3, inception_6a_5x5,
                                              inception_6a_pool_proj])

        inception_6b_1x1 = nlayers.conv2d(
            'inception_6b_1x1', inception_6a_output, 384, 1, 1)
        inception_6b_3x3_reduce = nlayers.conv2d(
            'inception_6b_3x3_reduce', inception_6a_output, 192, 1, 1)
        inception_6b_3x3 = nlayers.conv2d(
            'inception_6b_3x3', inception_6b_3x3_reduce, 384, 3, 1)
        inception_6b_5x5_reduce = nlayers.conv2d(
            'inception_6b_5x5_reduce', inception_6a_output, 48, 1, 1)
        inception_6b_5x5 = nlayers.conv2d(
            'inception_6b_5x5', inception_6b_5x5_reduce, 128, 5, 1)
        inception_6b_pool = nlayers.max_pool(
            'inception_6b_pool', inception_6a_output, 3, 1)
        inception_6b_pool_proj = nlayers.conv2d(
            'inception_6b_pool_proj', inception_6b_pool, 128, 1, 1)
        inception_6b_output = nlayers.concat('inception_6b_output',
                                             [inception_6b_1x1, inception_6b_3x3, inception_6b_5x5,
                                              inception_6b_pool_proj])
        pool6_drop_7x7_s1 = nlayers.dropout('pool6_drop_7x7_s1', inception_6b_output, 0.6)

        return [pool6_drop_7x7_s1, inception_5b_output, inception_4e_output, inception_3b_output]

    def fcn8s_decoder(self, tap_layers):
        """

        """
        fcn8s_output = None
        stddev_1x1 = 0.001  # Standard deviation for the 1x1 kernel initializers
        stddev_conv2d_trans = 0.01  # Standard deviation for the convolution transpose kernel initializers

        l2_regularization_rate = tf.placeholder(dtype=tf.float32, shape=[],
                                                name='l2_regularization_rate')  # L2 regularization rate for the kernels

        # first decode layer
        with tf.name_scope('decoder_0'):
            first_encode_layer = tap_layers[0]
            # Reduce channels to 19
            classifier_64s = tf.layers.conv2d(inputs=first_encode_layer,
                                              filters=self.config.n_classes,
                                              kernel_size=(1, 1),
                                              strides=(1, 1),
                                              padding='same',
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_1x1),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                  l2_regularization_rate),
                                              name='classifier_64s')
            # Upscale to 2x resolution
            upscale_32s = tf.layers.conv2d_transpose(inputs=classifier_64s,
                                                     filters=self.config.n_classes,
                                                     kernel_size=(4, 4),
                                                     strides=(2, 2),
                                                     padding='same',
                                                     kernel_initializer=tf.truncated_normal_initializer(
                                                         stddev=stddev_conv2d_trans),
                                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                         l2_regularization_rate),
                                                     name='upscale_32s')

            second_encode_layer = tap_layers[1]
            # Reduce channels to 19
            classifier_32s = tf.layers.conv2d(inputs=second_encode_layer,
                                              filters=self.config.n_classes,
                                              kernel_size=(1, 1),
                                              strides=(1, 1),
                                              padding='same',
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_1x1),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                  l2_regularization_rate),
                                              name='classifier_32s')

            # 2: fuse until we're back at the original image size.

            fuse_32s = tf.add(upscale_32s, classifier_32s, name='fuse_32s')

        # second decode layer
        with tf.name_scope('decoder_1'):
            # Upscale to 2x resolution
            upscale_16s = tf.layers.conv2d_transpose(inputs=fuse_32s,
                                                     filters=self.config.n_classes,
                                                     kernel_size=(4, 4),
                                                     strides=(2, 2),
                                                     padding='same',
                                                     kernel_initializer=tf.truncated_normal_initializer(
                                                         stddev=stddev_conv2d_trans),
                                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                         l2_regularization_rate),
                                                     name='upscale_16s')

            # Reduce channels to 19
            classifier_16s = tf.layers.conv2d(inputs=tap_layers[2],
                                              filters=self.config.n_classes,
                                              kernel_size=(1, 1),
                                              strides=(1, 1),
                                              padding='same',
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=stddev_1x1),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                  l2_regularization_rate),
                                              name='classifier_16s')

            # 2: fuse until we're back at the original image size.
            fuse_16s = tf.add(upscale_16s, classifier_16s, name='fuse_16s')

        # Third decode layer
        with tf.name_scope('decoder_2'):
            # Upscale to 2x resolution
            upscale_8s = tf.layers.conv2d_transpose(inputs=fuse_16s,
                                                    filters=self.config.n_classes,
                                                    kernel_size=(4, 4),
                                                    strides=(2, 2),
                                                    padding='same',
                                                    kernel_initializer=tf.truncated_normal_initializer(
                                                        stddev=stddev_conv2d_trans),
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                        l2_regularization_rate),
                                                    name='upscale_8s')

            # Reduce channels to 19
            classifier_8s = tf.layers.conv2d(inputs=tap_layers[3],
                                             filters=self.config.n_classes,
                                             kernel_size=(1, 1),
                                             strides=(1, 1),
                                             padding='same',
                                             kernel_initializer=tf.truncated_normal_initializer(
                                                 stddev=stddev_1x1),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                 l2_regularization_rate),
                                             name='classifier_8s')

            # 2: fuse until we're back at the original image size.
            fuse_8s = tf.add(upscale_8s, classifier_8s, name='fuse_8s')

        # Forth decode layer
        with tf.name_scope('decoder_3'):
            # Upscale to 2x resolution
            fcn8s_output = tf.layers.conv2d_transpose(inputs=fuse_8s,
                                                      filters=self.config.n_classes,
                                                      kernel_size=(16, 18),
                                                      strides=(8, 8),
                                                      padding='same',
                                                      kernel_initializer=tf.truncated_normal_initializer(
                                                          stddev=stddev_conv2d_trans),
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                          l2_regularization_rate),
                                                      name='fcn8s_output')

        return fcn8s_output

    def build_model(self):
        _corenet_tap = self.core_net()
        self.logits = self.fcn8s_decoder(_corenet_tap)

    def init_variables(self):
        return tf.initialize_all_variables()


if __name__ == "__main__":
    config = networkConfig()
    print(config.batch_size)
    _model = Inceptionv1FCN8s(config)
    _model.build_model()

    saver = tf.train.Saver()

    _data = np.ones(shape=(config.batch_size, config.image_width,
                           config.image_height, config.image_depth), dtype=np.float32)
    _labels = np.ones(shape=(config.batch_size, config.image_width,
                             config.image_height, config.n_classes), dtype=np.float32)

    init_op = _model.init_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        logits = sess.run(_model.logits, feed_dict={_model.in_data: _data, _model.labels: _labels})
        # Save the variables to disk.
        save_path = saver.save(sess, "./test_output/model.ckpt")
        print("Model saved in path: %s" % save_path)

        model_vars = tf.trainable_variables()

        for ivar in model_vars:
            print(ivar)
