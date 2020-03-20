import tensorflow as tf
import numpy as np

import layers


class InceptionV1:
    def __init__(self, config):
        self.in_data = tf.placeholder(tf.float32, name='in_data', shape=[config.batch_size,
                                                                         config.image_width,
                                                                         config.image_height,
                                                                         config.image_depth], )
        self.logits = tf.placeholder(tf.int16, name='logits', shape=[config.batch_size,
                                                                     config.n_classes])

    def net(self):
        conv1_7x7_s2 = layers.conv2d('conv1_7x7_s2', self.in_data, 64, 7, 2)
        pool1_3x3_s2 = layers.max_pool('pool1_3x3_s2', conv1_7x7_s2, 3, 2)
        pool1_norm1 = layers.lrn('pool1_norm1', pool1_3x3_s2)
        conv2_3x3_reduce = layers.conv2d(
            'conv2_3x3_reduce', pool1_norm1, 64, 1, 1)
        conv2_3x3 = layers.conv2d('conv2_3x3', conv2_3x3_reduce, 192, 3, 1)
        conv2_norm2 = layers.lrn('conv2_norm2', conv2_3x3)
        pool2_3x3_s2 = layers.max_pool('pool2_3x3_s2', conv2_norm2, 3, 2)

        inception_3a_1x1 = layers.conv2d(
            'inception_3a_1x1', pool2_3x3_s2, 64, 1, 1)
        inception_3a_3x3_reduce = layers.conv2d(
            'inception_3a_3x3_reduce', pool2_3x3_s2, 96, 1, 1)
        inception_3a_3x3 = layers.conv2d(
            'inception_3a_3x3', inception_3a_3x3_reduce, 128, 3, 1)
        inception_3a_5x5_reduce = layers.conv2d(
            'inception_3a_5x5_reduce', pool2_3x3_s2, 16, 1, 1)
        inception_3a_5x5 = layers.conv2d(
            'inception_3a_5x5', inception_3a_5x5_reduce, 32, 5, 1)
        inception_3a_pool = layers.max_pool(
            'inception_3a_pool', pool2_3x3_s2, 3, 1)
        inception_3a_pool_proj = layers.conv2d(
            'inception_3a_pool_proj', inception_3a_pool, 32, 1, 1)
        inception_3a_output = layers.concat('inception_3a_output', [inception_3a_1x1, inception_3a_3x3, inception_3a_5x5,
                                                                    inception_3a_pool_proj])

        inception_3b_1x1 = layers.conv2d(
            'inception_3b_1x1', inception_3a_output, 128, 1, 1)
        inception_3b_3x3_reduce = layers.conv2d(
            'inception_3b_3x3_reduce', inception_3a_output, 128, 1, 1)
        inception_3b_3x3 = layers.conv2d(
            'inception_3b_3x3', inception_3b_3x3_reduce, 192, 3, 1)
        inception_3b_5x5_reduce = layers.conv2d(
            'inception_3b_5x5_reduce', inception_3a_output, 32, 1, 1)
        inception_3b_5x5 = layers.conv2d(
            'inception_3b_5x5', inception_3b_5x5_reduce, 96, 5, 1)
        inception_3b_pool = layers.max_pool(
            'inception_3b_pool', inception_3a_output, 3, 1)
        inception_3b_pool_proj = layers.conv2d(
            'inception_3b_pool_proj', inception_3b_pool, 64, 1, 1)
        inception_3b_output = layers.concat('inception_3b_output', [inception_3b_1x1, inception_3b_3x3, inception_3b_5x5,
                                                                    inception_3b_pool_proj])

        pool3_3x3_s2 = layers.max_pool(
            'pool3_3x3_s2', inception_3b_output, 3, 2)
        inception_4a_1x1 = layers.conv2d(
            'inception_4a_1x1', pool3_3x3_s2, 192, 1, 1)
        inception_4a_3x3_reduce = layers.conv2d(
            'inception_4a_3x3_reduce', pool3_3x3_s2, 96, 1, 1)
        inception_4a_3x3 = layers.conv2d(
            'inception_4a_3x3', inception_4a_3x3_reduce, 208, 3, 1)
        inception_4a_5x5_reduce = layers.conv2d(
            'inception_4a_5x5_reduce', pool3_3x3_s2, 16, 1, 1)
        inception_4a_5x5 = layers.conv2d(
            'inception_4a_5x5', inception_4a_5x5_reduce, 48, 5, 1)
        inception_4a_pool = layers.max_pool(
            'inception_4a_pool', pool3_3x3_s2, 3, 1)
        inception_4a_pool_proj = layers.conv2d(
            'inception_4a_pool_proj', inception_4a_pool, 64, 1, 1)
        inception_4a_output = layers.concat('inception_4a_output', [inception_4a_1x1, inception_4a_3x3, inception_4a_5x5,
                                                                    inception_4a_pool_proj])

        inception_4b_1x1 = layers.conv2d(
            'inception_4b_1x1', inception_4a_output, 160, 1, 1)
        inception_4b_3x3_reduce = layers.conv2d(
            'inception_4b_3x3_reduce', inception_4a_output, 112, 1, 1)
        inception_4b_3x3 = layers.conv2d(
            'inception_4b_3x3', inception_4b_3x3_reduce, 224, 3, 1)
        inception_4b_5x5_reduce = layers.conv2d(
            'inception_4b_5x5_reduce', inception_4a_output, 24, 1, 1)
        inception_4b_5x5 = layers.conv2d(
            'inception_4b_5x5', inception_4b_5x5_reduce, 64, 5, 1)
        inception_4b_pool = layers.max_pool(
            'inception_4b_pool', inception_4a_output, 3, 1)
        inception_4b_pool_proj = layers.conv2d(
            'inception_4b_pool_proj', inception_4b_pool, 64, 1, 1)
        inception_4b_output = layers.concat('inception_4b_output', [inception_4b_1x1, inception_4b_3x3, inception_4b_5x5,
                                                                    inception_4b_pool_proj])

        inception_4c_1x1 = layers.conv2d(
            'inception_4c_1x1', inception_4b_output, 128, 1, 1)
        inception_4c_3x3_reduce = layers.conv2d(
            'inception_4c_3x3_reduce', inception_4b_output, 128, 1, 1)
        inception_4c_3x3 = layers.conv2d(
            'inception_4c_3x3', inception_4c_3x3_reduce, 256, 3, 1)
        inception_4c_5x5_reduce = layers.conv2d(
            'inception_4c_5x5_reduce', inception_4b_output, 24, 1, 1)
        inception_4c_5x5 = layers.conv2d(
            'inception_4c_5x5', inception_4c_5x5_reduce, 64, 5, 1)
        inception_4c_pool = layers.max_pool(
            'inception_4c_pool', inception_4b_output, 3, 1)
        inception_4c_pool_proj = layers.conv2d(
            'inception_4c_pool_proj', inception_4c_pool, 64, 1, 1)
        inception_4c_output = layers.concat('inception_4c_output', [inception_4c_1x1, inception_4c_3x3, inception_4c_5x5,
                                                                    inception_4c_pool_proj])

        inception_4d_1x1 = layers.conv2d(
            'inception_4d_1x1', inception_4c_output, 112, 1, 1)
        inception_4d_3x3_reduce = layers.conv2d(
            'inception_4d_3x3_reduce', inception_4c_output, 144, 1, 1)
        inception_4d_3x3 = layers.conv2d(
            'inception_4d_3x3', inception_4d_3x3_reduce, 288, 3, 1)
        inception_4d_5x5_reduce = layers.conv2d(
            'inception_4d_5x5_reduce', inception_4c_output, 32, 1, 1)
        inception_4d_5x5 = layers.conv2d(
            'inception_4d_5x5', inception_4d_5x5_reduce, 64, 5, 1)
        inception_4d_pool = layers.max_pool(
            'inception_4d_pool', inception_4c_output, 3, 1)
        inception_4d_pool_proj = layers.conv2d(
            'inception_4d_pool_proj', inception_4d_pool, 64, 1, 1)
        inception_4d_output = layers.concat('inception_4d_output', [inception_4d_1x1, inception_4d_3x3, inception_4d_5x5,
                                                                    inception_4d_pool_proj])

        inception_4e_1x1 = layers.conv2d(
            'inception_4e_1x1', inception_4d_output, 256, 1, 1)
        inception_4e_3x3_reduce = layers.conv2d(
            'inception_4e_3x3_reduce', inception_4d_output, 160, 1, 1)
        inception_4e_3x3 = layers.conv2d(
            'inception_4e_3x3', inception_4e_3x3_reduce, 320, 3, 1)
        inception_4e_5x5_reduce = layers.conv2d(
            'inception_4e_5x5_reduce', inception_4d_output, 32, 1, 1)
        inception_4e_5x5 = layers.conv2d(
            'inception_4e_5x5', inception_4e_5x5_reduce, 128, 5, 1)
        inception_4e_pool = layers.max_pool(
            'inception_4e_pool', inception_4d_output, 3, 1)
        inception_4e_pool_proj = layers.conv2d(
            'inception_4e_pool_proj', inception_4e_pool, 128, 1, 1)
        inception_4e_output = layers.concat('inception_4e_output', [inception_4e_1x1, inception_4e_3x3, inception_4e_5x5,
                                                                    inception_4e_pool_proj])

        pool4_3x3_s2 = layers.max_pool(
            'pool4_3x3_s2', inception_4e_output, 3, 2)
        inception_5a_1x1 = layers.conv2d(
            'inception_5a_1x1', pool4_3x3_s2, 256, 1, 1)
        inception_5a_3x3_reduce = layers.conv2d(
            'inception_5a_3x3_reduce', pool4_3x3_s2, 160, 1, 1)
        inception_5a_3x3 = layers.conv2d(
            'inception_5a_3x3', inception_5a_3x3_reduce, 320, 3, 1)
        inception_5a_5x5_reduce = layers.conv2d(
            'inception_5a_5x5_reduce', pool4_3x3_s2, 32, 1, 1)
        inception_5a_5x5 = layers.conv2d(
            'inception_5a_5x5', inception_5a_5x5_reduce, 128, 5, 1)
        inception_5a_pool = layers.max_pool(
            'inception_5a_pool', pool4_3x3_s2, 3, 1)
        inception_5a_pool_proj = layers.conv2d(
            'inception_5a_pool_proj', inception_5a_pool, 128, 1, 1)
        inception_5a_output = layers.concat('inception_5a_output', [inception_5a_1x1, inception_5a_3x3, inception_5a_5x5,
                                                                    inception_5a_pool_proj])

        inception_5b_1x1 = layers.conv2d(
            'inception_5b_1x1', inception_5a_output, 384, 1, 1)
        inception_5b_3x3_reduce = layers.conv2d(
            'inception_5b_3x3_reduce', inception_5a_output, 192, 1, 1)
        inception_5b_3x3 = layers.conv2d(
            'inception_5b_3x3', inception_5b_3x3_reduce, 384, 3, 1)
        inception_5b_5x5_reduce = layers.conv2d(
            'inception_5b_5x5_reduce', inception_5a_output, 48, 1, 1)
        inception_5b_5x5 = layers.conv2d(
            'inception_5b_5x5', inception_5b_5x5_reduce, 128, 5, 1)
        inception_5b_pool = layers.max_pool(
            'inception_5b_pool', inception_5a_output, 3, 1)
        inception_5b_pool_proj = layers.conv2d(
            'inception_5b_pool_proj', inception_5b_pool, 128, 1, 1)
        inception_5b_output = layers.concat('inception_5b_output', [inception_5b_1x1, inception_5b_3x3, inception_5b_5x5,
                                                                    inception_5b_pool_proj])

        pool5_7x7_s1 = layers.avg_pool(
            'pool5_7x7_s1', inception_5b_output, 7, 1)
        pool5_drop_7x7_s1 = layers.dropout(
            'pool5_drop_7x7_s1', pool5_7x7_s1, 0.6)


if __name__ == "__main__":
    config = layers.Config()
    _model = InceptionV1(config)
    _model.net()
