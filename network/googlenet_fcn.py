from distutils.version import LooseVersion
import tensorflow as tf
import warnings
from tqdm import trange
import sys
import os.path
import scipy.misc
import shutil
from glob import glob
from collections import deque
import numpy as np
import time

from helpers.tf_variable_summaries import add_variable_summaries
from helpers.visualization_utils import print_segmentation_onto_image, create_split_view

import network.nlayers as nlayers
from network.nlayers import networkConfig


class Inceptionv1FCN8s:
    def __init__(self, experiment_dir, exp_config, model_load_dir=None, tags=None, variables_load_dir=None):
        """
                Arguments:
                    model_load_dir (string, optional): The directory path to a `SavedModel`, i.e. to the directory
                        that contains a saved FCN-8s model protocol buffer. If a path is provided, the targeted model will
                        be loaded. If no path is given, the model will be built from scratch on top of a pre-trained,
                        convolutionalized VGG-16 base network. `model_load_dir` and `vgg16_dir` may not both be `None`.
                    tags (list, optional): Only relevant if a path to a saved FCN-8s model is given in `model_load_dir`.
                        A list of strings containing the tags required to load the appropriate metagraph.
                    vgg16_dir (string, optional): Only relevant if no path to a saved FCN-8s model is given in `model_load_dir`.
                        The directory that contains a pretrained, convolutionalized VGG-16 model in the form of a protocol buffer.
                        `model_load_dir` and `vgg16_dir` may not both be `None`.
                    num_classes (int, optional): Only relevant if no path to a saved FCN-8s model is given in `model_load_dir`.
                        The number of segmentation classes.
                    variables_load_dir (string, optional): The path to variables that were saved with `tf.train.Saver`.
                        Only relevant if `model_load_dir` is `None`.
                """
        # Check TensorFlow version
        assert LooseVersion(tf.__version__) >= LooseVersion(
            '1.0'), 'This program requires TensorFlow version 1.0 or newer. You are using {}'.format(tf.__version__)
        print('TensorFlow Version: {}'.format(tf.__version__))

        self.experiment_dir = experiment_dir
        self.variables_load_dir = variables_load_dir
        self.model_load_dir = model_load_dir
        self.tags = tags
        self.num_classes = exp_config.n_classes

        self.variables_updated = False  # Keep track of whether any variable values changed since this model was last saved.
        self.eval_dataset = None  # Which dataset to use for evaluation during training. Only relevant for training.

        # The following lists store data about the metrics being tracked.
        # Note that `self.metric_value_tensors` and `self.metric_update_ops` represent
        # the metrics being tracked, not the metrics generally available in the model.
        self.metric_names = []  # Store the metric names here.
        self.metric_values = []  # Store the latest metric evaluations here.
        self.best_metric_values = []  # Keep score of the best historical metric values.
        self.metric_value_tensors = []  # Store the value tensors from tf.metrics here.
        self.metric_update_ops = []  # Store the update ops from tf.metrics here.

        self.training_loss = None
        self.best_training_loss = 99999999.9

        self.sess = tf.Session()
        self.g_step = None  # The global step
        self.image_input = tf.placeholder(tf.float32, name='in_data', shape=[None, None, None, 3], )
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        # Build the part of the graph that is relevant for the training.
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None, None, self.num_classes],
                                     name='labels_input')
        self.config = exp_config

        self.fcn8s_output, self.l2_regularization_rate = self._build_model()
        self.total_loss, self.train_op, self.learning_rate, self.global_step = self._build_optimizer()

        # Add the prediction outputs.
        self.softmax_output, self.predictions_argmax = self._build_predictor()

        # Add metrics for evaluation.
        self.mean_loss_value, self.mean_loss_update_op, self.mean_iou_value, self.mean_iou_update_op, self.acc_value, self.acc_update_op, self.metrics_reset_op = self._build_metrics()

        # Add summary ops for TensorBoard.
        self.summaries_training, self.summaries_evaluation = self._build_summary_ops()

        # Initialize the global and local (for the metrics) variables.
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def _core_net(self):
        conv1_7x7_s2 = nlayers.conv2d('conv1_7x7_s2', self.image_input, 64, 7, 2)
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
        pool6_drop_7x7_s1 = nlayers.dropout('pool6_drop_7x7_s1', inception_6b_output, keep_prob=self.keep_prob)

        return [pool6_drop_7x7_s1, inception_5b_output, inception_4e_output, inception_3b_output]

    def _build_decoder(self, tap_layers):
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
            # Upscale to 2x resolution
            upscale_1s = tf.layers.conv2d_transpose(inputs=fuse_8s,
                                                    filters=self.config.n_classes,
                                                    kernel_size=(16, 18),
                                                    strides=(8, 8),
                                                    padding='same',
                                                    kernel_initializer=tf.truncated_normal_initializer(
                                                        stddev=stddev_conv2d_trans),
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                        l2_regularization_rate),
                                                    name='upscale_1s')
            fcn8s_output = tf.identity(upscale_1s, name='fcn8s_output')
        return fcn8s_output, l2_regularization_rate

    def _build_optimizer(self):
        """
        Builds the training-relevant part of the graph.
        """

        with tf.name_scope('optimizer'):
            # Create a training step counter.
            global_step = tf.Variable(0, trainable=False, name='global_step')
            # Create placeholder for the learning rate.
            learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
            # Compute the regularizatin loss.
            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)  # This is a list of the individual loss values, so we still need to sum them up.
            regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')  # Scalar
            # Compute the total loss.
            approximation_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.fcn8s_output),
                name='approximation_loss')  # Scalar
            total_loss = tf.add(approximation_loss, regularization_loss, name='total_loss')
            # Compute the gradients and apply them.
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam_optimizer')
            train_op = optimizer.minimize(total_loss, global_step=global_step, name='train_op')

        return total_loss, train_op, learning_rate, global_step

    def _build_predictor(self):
        '''
        Builds the prediction-relevant part of the graph.
        '''

        with tf.name_scope('predictor'):
            softmax_output = tf.nn.softmax(self.fcn8s_output, name='softmax_output')
            predictions_argmax = tf.argmax(softmax_output, axis=-1, name='predictions_argmax', output_type=tf.int64)

        return softmax_output, predictions_argmax

    def _build_model(self):
        _corenet_tap = self._core_net()
        fcn8s_output, l2_regularization_rate = self._build_decoder(_corenet_tap)
        return fcn8s_output, l2_regularization_rate

    def init_variables(self):
        return tf.initialize_all_variables()

    def _build_metrics(self):
        '''
        Builds the evaluation-relevant part of the graph, i.e. the metrics operations.
        '''

        with tf.variable_scope('metrics') as scope:
            labels_argmax = tf.argmax(self.labels, axis=-1, name='labels_argmax', output_type=tf.int64)

            # 1: Mean loss

            mean_loss_value, mean_loss_update_op = tf.metrics.mean(self.total_loss)

            mean_loss_value = tf.identity(mean_loss_value, name='mean_loss_value')
            mean_loss_update_op = tf.identity(mean_loss_update_op, name='mean_loss_update_op')

            # 1: Mean IoU

            mean_iou_value, mean_iou_update_op = tf.metrics.mean_iou(labels=labels_argmax,
                                                                     predictions=self.predictions_argmax,
                                                                     num_classes=self.num_classes)

            mean_iou_value = tf.identity(mean_iou_value, name='mean_iou_value')
            mean_iou_update_op = tf.identity(mean_iou_update_op, name='mean_iou_update_op')

            # 2: Accuracy

            acc_value, acc_update_op = tf.metrics.accuracy(labels=labels_argmax,
                                                           predictions=self.predictions_argmax)

            acc_value = tf.identity(acc_value, name='acc_value')
            acc_update_op = tf.identity(acc_update_op, name='acc_update_op')

            # As of version 1.3, TensorFlow's streaming metrics don't have reset operations,
            # so we need to create our own as a work-around. Say we want to evaluate
            # a metric after every training epoch. If we didn't have
            # a way to reset the metric's update op after every evaluation,
            # the computed metric value would be the average of the current evaluation
            # and all previous evaluations from past epochs, which is obviously not
            # what we want.
            local_metric_vars = tf.contrib.framework.get_variables(scope=scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            metrics_reset_op = tf.variables_initializer(var_list=local_metric_vars, name='metrics_reset_op')

        return (mean_loss_value,
                mean_loss_update_op,
                mean_iou_value,
                mean_iou_update_op,
                acc_value,
                acc_update_op,
                metrics_reset_op)

    def _build_summary_ops(self):
        """
        Builds the part of the graph that logs summaries for TensorBoard.
        """

        graph = tf.get_default_graph()

        add_variable_summaries(variable=graph.get_tensor_by_name('inception_3b_pool_proj/weights:0'),
                               scope='inception_3b_pool_proj')
        add_variable_summaries(variable=graph.get_tensor_by_name('inception_3b_pool_proj/biases:0'),
                               scope='inception_3b_pool_proj')

        # Loss and learning rate.
        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('learning_rate', self.learning_rate)

        summaries_training = tf.summary.merge_all()
        summaries_training = tf.identity(summaries_training, name='summaries_training')

        # All metrics.
        mean_loss = tf.summary.scalar('mean_loss', self.mean_loss_value)
        mean_iou = tf.summary.scalar('mean_iou', self.mean_iou_value)
        accuracy = tf.summary.scalar('accuracy', self.acc_value)

        summaries_evaluation = tf.summary.merge(inputs=[mean_loss,
                                                        mean_iou,
                                                        accuracy])
        summaries_evaluation = tf.identity(summaries_evaluation, name='summaries_evaluation')

        return summaries_training, summaries_evaluation

    def _initialize_metrics(self, metrics):
        """
        Initializes/resets the metrics before every call to `train` and `evaluate`.
        """

        # Reset lists of previous tracked metrics.
        self.metric_names = []
        self.best_metric_values = []
        self.metric_update_ops = []
        self.metric_value_tensors = []

        # Set the metrics that will be evaluated.
        if 'loss' in metrics:
            self.metric_names.append('loss')
            self.best_metric_values.append(99999999.9)
            self.metric_update_ops.append(self.mean_loss_update_op)
            self.metric_value_tensors.append(self.mean_loss_value)
        if 'mean_iou' in metrics:
            self.metric_names.append('mean_iou')
            self.best_metric_values.append(0.0)
            self.metric_update_ops.append(self.mean_iou_update_op)
            self.metric_value_tensors.append(self.mean_iou_value)
        if 'accuracy' in metrics:
            self.metric_names.append('accuracy')
            self.best_metric_values.append(0.0)
            self.metric_update_ops.append(self.acc_update_op)
            self.metric_value_tensors.append(self.acc_value)

    def train(self,
              train_generator,
              epochs,
              steps_per_epoch,
              learning_rate_schedule,
              keep_prob=0.5,
              l2_regularization=0.0,
              eval_dataset='train',
              eval_frequency=5,
              val_generator=None,
              val_steps=None,
              metrics={},
              save_during_training=False,
              save_dir=None,
              save_best_only=True,
              save_tags=['default'],
              save_name='',
              save_frequency=5,
              saver='saved_model',
              monitor='loss',
              record_summaries=True,
              summaries_frequency=10,
              summaries_dir=None,
              summaries_name=None,
              training_loss_display_averaging=3):
        """
        Trains the model.

        Arguments:
            train_generator (generator): A generator that yields batches of images
                and associated ground truth images in two separate Numpy arrays.
                The images must be a 4D array with format `(batch_size, height, width, channels)`
                and the ground truth images must be a 4D array with format
                `(batch_size, height, width, num_classes)`, i.e. the ground truth
                data must be provided in one-hot format.
            epochs (int): The number of epochs to run the training for, where each epoch
                consists of `steps_per_epoch` training steps.
            steps_per_epoch (int): The number of training steps (i.e. batches processed)
                per epoch.
            learning_rate_schedule (function): Any function that takes as its sole input
                an integer (the global step counter) and returns a float (the learning rate).
            keep_prob (float, optional): The keep probability for the two dropout layers
                in the VGG-16 encoder network. Defaults to 0.5.
            l2_regularization (float, optional): The scaling factor for the L2 regularization
                of all decoder kernels. 0 means no regularization at all. This has no effect
                on the kernels of the VGG-16 encoder network. Defaults to 0.
            eval_dataset (string, optional): Which generator to use for the evaluation
                of the model during training. Can be either of 'train' (the train_generator
                will be used) or 'val' (the val_generator will be used). Defaults to 'train',
                but should be set to 'val' if a validation dataset is available.
            eval_frequency (int, optional): The model will be evaluated on `metrics` after every
                `eval_frequency` epochs. Defaults to 5.
            val_generator (generator, optional): An optional second generator for a second
                dataset (validation dataset), works the same way as `train_generator`.
            val_steps (int, optional): The number of steps to run `val_generator` for
                during evaluation.
            metrics (set, optional): The metrics to be evaluated during training. A Python
                set containing any subset of `{'loss', 'mean_iou', 'accuracy'}`, which are the
                currently available metrics. Defaults to the empty set, meaning that the
                model will not be evaluated during training.
            save_during_training (bool, optional): Whether or not to save the model periodically
                during training, the parameters of which can be set in the subsequent arguments.
                Defaults to `False`.
            save_dir (string, optional): The full path of the directory to save the model to
                during training.
            save_best_only (bool, optional): If `True`, the model will only be saved upon
                evaluation if the metric defined by `monitor` has improved since it was last
                measured before. Can only be `True` if `metrics` is not empty.
            save_tags (list, optional): An optional list of tags to save the model metagraph
                with in the SavedModel protocol buffer. Defaults to a list only containing
                the tag 'default'. At least one tag must be given.
            save_name (string, optional): An optional name string to include in the name of
                the folder in which the model will be saved during training. Note that what
                you pass as the name here will be only part of the folder name. The folder
                name also includes a count of the global training step and the values of
                any metrics that are being evaluate, although at least the training loss.
                It is hence not necessary to pass a name here, each saved model will be
                uniquely and descriptively named regardless. Defaults to the empty string.
            save_frequency (int, optional): The model will be saved at most after every
                `save_frequency` epochs, but possibly less often if `save_best_only` is `True`
                and if there was no improvement in the monitored metric. Defaults to 5.
            saver (string, optional): Which saver to use when saving the model during training.
                Can be either of 'saved_model' in order to use `tf.saved_model` or 'train_saver'
                in order to use `tf.train.Saver`. Defaults to `tf.saved_model`. Check the
                TensorFlow documentation for details on which saver might be better for your
                use case. In general you can't go wrong with either of the two.
            monitor (string, optional): The name of the metric that is to be monitored in
                order to decide whether the model should be saved. Can be one of
                `{'loss', 'mean_iou', 'accuracy'}`, which are the currently available metrics.
                Defaults to 'loss'.
            record_summaries (bool, optional): Whether or not to record TensorBoard summaries.
                Defaults to `True`.
            summaries_frequency (int, optional): How often summaries should be logged for
                tensors which are updated at every training step. The summaries for such tensors
                will be recorded every `summaries_frequency` training steps. Defaults to 10.
            summaries_dir (string, optional): The full path of the directory to which to
                write the summaries protocol buffers.
            summaries_name (string, optional): The name of the summaries buffers.
            training_loss_display_averaging (int, optional): During training, the current
                training loss is always displayed. Since training on mini-batches has the effect
                that the loss might jump from training step to training step, this parameter
                allows to average the displayed loss over tha lasst `training_loss_display_averaging`
                training steps so that it shows a more representative picture of the actual
                current loss. Defaults to 3.
        """

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please note that training this network will be unbearably slow without a GPU.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

        if not eval_dataset in ['train', 'val']:
            raise ValueError("`eval_dataset` must be one of 'train' or 'val', but is '{}'.".format(eval_dataset))

        if (eval_dataset == 'val') and ((val_generator is None) or (val_steps is None)):
            raise ValueError("When eval_dataset == 'val', a `val_generator` and `val_steps` must be passed.")

        for metric in metrics:
            if not metric in ['loss', 'mean_iou', 'accuracy']:
                raise ValueError(
                    "{} is not a valid metric. Valid metrics are ['loss', mean_iou', 'accuracy']".format(metric))

        if (not monitor in metrics) and (not monitor == 'loss'):
            raise ValueError(
                'You are trying to monitor {}, but it is not in `metrics` and is therefore not being computed.'.format(
                    monitor))

        self.eval_dataset = eval_dataset

        self.g_step = self.sess.run(self.global_step)
        learning_rate = learning_rate_schedule(self.g_step)

        self._initialize_metrics(metrics)

        # Set up the summary file writers.
        if record_summaries:
            training_writer = tf.summary.FileWriter(logdir=os.path.join(summaries_dir, summaries_name),
                                                    graph=self.sess.graph)
            if len(metrics) > 0:
                evaluation_writer = tf.summary.FileWriter(logdir=os.path.join(summaries_dir, summaries_name + '_eval'))

        for epoch in range(1, epochs + 1):

            ##############################################################
            # Run the training for this epoch.
            ##############################################################

            loss_history = deque(maxlen=training_loss_display_averaging)

            tr = trange(steps_per_epoch, file=sys.stdout)
            tr.set_description('Epoch {}/{}'.format(epoch, epochs))

            # for train_step in tr:
            for train_step in range(5):

                batch_images, batch_labels = next(train_generator)

                if record_summaries and (self.g_step % summaries_frequency == 0):
                    _, current_loss, self.g_step, training_summary = self.sess.run([self.train_op,
                                                                                    self.total_loss,
                                                                                    self.global_step,
                                                                                    self.summaries_training],
                                                                                   feed_dict={
                                                                                       self.image_input: batch_images,
                                                                                       self.labels: batch_labels,
                                                                                       self.learning_rate: learning_rate,
                                                                                       self.keep_prob: keep_prob,
                                                                                       self.l2_regularization_rate: l2_regularization})
                    training_writer.add_summary(summary=training_summary, global_step=self.g_step)
                else:
                    _, current_loss, self.g_step = self.sess.run([self.train_op,
                                                                  self.total_loss,
                                                                  self.global_step],
                                                                 feed_dict={self.image_input: batch_images,
                                                                            self.labels: batch_labels,
                                                                            self.learning_rate: learning_rate,
                                                                            self.keep_prob: keep_prob,
                                                                            self.l2_regularization_rate: l2_regularization})

                self.variables_updated = True

                loss_history.append(current_loss)
                losses = np.array(loss_history)
                self.training_loss = np.mean(losses)

                tr.set_postfix(ordered_dict={'loss': self.training_loss,
                                             'learning rate': learning_rate})

                learning_rate = learning_rate_schedule(self.g_step)

            ##############################################################
            # Maybe evaluate the model after this epoch.
            ##############################################################

            if (len(metrics) > 0) and (epoch % eval_frequency == 0):

                if eval_dataset == 'train':
                    data_generator = train_generator
                    num_batches = steps_per_epoch
                    description = 'Evaluation on training dataset'
                elif eval_dataset == 'val':
                    data_generator = val_generator
                    num_batches = val_steps
                    description = 'Evaluation on validation dataset'

                self._evaluate(eval_dir='{}/cityscapes_eval'.format(self.experiment_dir),
                               data_generator=data_generator,
                               metrics=metrics,
                               num_batches=num_batches,
                               l2_regularization=l2_regularization,
                               description=description)

                if record_summaries:
                    evaluation_summary = self.sess.run(self.summaries_evaluation)
                    evaluation_writer.add_summary(summary=evaluation_summary, global_step=self.g_step)

            ##############################################################
            # Maybe save the model after this epoch.
            ##############################################################

            if save_during_training and (epoch % save_frequency == 0):

                save = False
                if save_best_only:
                    if (monitor == 'loss' and
                            (not 'loss' in self.metric_names) and
                            self.training_loss < self.best_training_loss):
                        save = True
                    else:
                        i = self.metric_names.index(monitor)
                        if (monitor == 'loss') and (self.metric_values[i] < self.best_metric_values[i]):
                            save = True
                        elif (monitor in ['accuracry', 'mean_iou']) and (
                                self.metric_values[i] > self.best_metric_values[i]):
                            save = True
                    if save:
                        print('New best {} value, saving model.'.format(monitor))
                    else:
                        print('No improvement over previous best {} value, not saving model.'.format(monitor))
                else:
                    save = True

                if save:
                    self.save(model_save_dir=save_dir,
                              saver=saver,
                              tags=save_tags,
                              name=save_name,
                              include_global_step=True,
                              include_last_training_loss=True,
                              include_metrics=(len(self.metric_names) > 0))

            ##############################################################
            # Update the current best metric values.
            ##############################################################

            if self.training_loss < self.best_training_loss:
                self.best_training_loss = self.training_loss

            if epoch % eval_frequency == 0:

                for i, metric_name in enumerate(self.metric_names):
                    if (metric_name == 'loss') and (self.metric_values[i] < self.best_metric_values[i]):
                        self.best_metric_values[i] = self.metric_values[i]
                    elif (metric_name in ['accuracry', 'mean_iou']) and (
                            self.metric_values[i] > self.best_metric_values[i]):
                        self.best_metric_values[i] = self.metric_values[i]

    def _evaluate(self, eval_dir, data_generator, metrics, num_batches, l2_regularization,
                  description='Running evaluation'):
        """
        Internal method used by both `evaluate()` and `train()` that performs
        the actual evaluation. For the first three arguments, please refer
        to the documentation of the public `evaluate()` method.

        Arguments:
            description (string, optional): A description string that will be prepended
                to the progress bar while the evaluation is being processed. During
                training, this description is used to clarify whether the evaluation
                is being performed on the training or validation dataset.
        """

        # Reset all metrics' accumulator variables.
        self.sess.run(self.metrics_reset_op)

        # Set up the progress bar.
        tr = trange(num_batches, file=sys.stdout)
        tr.set_description(description)

        # Accumulate metrics in batches.
        # for step in tr:
        for step in range(5):
            batch_images, batch_labels = next(data_generator)

            self.sess.run(self.metric_update_ops,
                          feed_dict={self.image_input: batch_images,
                                     self.labels: batch_labels,
                                     self.keep_prob: 1.0,
                                     self.l2_regularization_rate: l2_regularization})

        # Compute final metric values.
        self.metric_values = self.sess.run(self.metric_value_tensors)

        evaluation_results_string = ''
        for i, metric_name in enumerate(self.metric_names):
            evaluation_results_string += metric_name + ': {:.4f}  '.format(self.metric_values[i])
        print(evaluation_results_string)

        # Save Evaluation in a file
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        evaluation_results_filename = os.path.join(eval_dir, "eval_report_{}.txt".format(self.g_step))
        with open(evaluation_results_filename, "w+") as fptr:
            fptr.write(evaluation_results_string)

    def evaluate(self, eval_dir, data_generator, num_batches, metrics={'loss', 'mean_iou', 'accuracy'},
                 l2_regularization=0.0, dataset='val'):
        """
        Evaluates the model on the given metrics on the data generated by `data_generator`.

        Arguments:
            data_generator (generator): A generator that yields batches of images
                and associated ground truth images in two separate Numpy arrays.
                The images must be a 4D array with format `(batch_size, height, width, channels)`
                and the ground truth images must be a 4D array with format
                `(batch_size, height, width, num_classes)`, i.e. the ground truth
                data must be provided in one-hot format. The generator's batch size
                has no effect on the outcome of the evaluation.
            num_batches (int): The number of batches to evaluate the model on.
                Typically this will be the number of batches such that the model
                is being evaluated on the whole evaluation dataset.
            metrics (set, optional): The metrics to be evaluated. A Python set containing
                any subset of `{'loss', 'mean_iou', 'accuracy'}`, which are the
                currently available metrics. Defaults to the full set.
            dataset (string, optional): Specifies the kind of dataset on which the model
                is being evaluated. Should be set to 'train' if the model is being evaluated
                on a dataset on which it has also been trained, or 'val' if the model is
                being evaluated on a dataset which it has never seen during training.
                This argument has no effect on the evaluation of the model, but if you
                save the model using `save()` after evaluating it, the model name will
                include this value to indicate whether or not the metric values were
                achieved on a dataset that has not been used during training. Defaults to 'val'.
            l2_regularization
        """

        for metric in metrics:
            if not metric in ['loss', 'mean_iou', 'accuracy']:
                raise ValueError(
                    "{} is not a valid metric. Valid metrics are ['loss', mean_iou', 'accuracy']".format(metric))

        if not dataset in {'train', 'val'}:
            raise ValueError("`dataset` must be either 'train' or 'val'.")

        self._initialize_metrics(metrics)

        self._evaluate(eval_dir, data_generator, metrics, num_batches, l2_regularization,
                       description='Running evaluation')

        if dataset == 'val':
            self.eval_dataset = 'val'
        else:
            self.eval_dataset = 'train'

    def predict(self, images, argmax=True):
        """
        Makes predictions for the input images.

        Arguments:
            images (array-like): The input image or images. Must be an array-like
                object of rank 4. If predicting only one image, encapsulate it in
                a Python list.
            argmax (bool, optional): If `True`, the model predicts class IDs,
                i.e. the last dimension has length 1 and an integer between
                zero and `num_classes - 1` for each pixel. Otherwise, the model
                outputs the softmax distribution, i.e. the last dimension has
                length `num_classes` and contains the probability for each class
                for all pixels. Defaults to `True`.

        Returns:
            The prediction, an array of rank 4 of which the first three dimensions
            are identical to the input and the fourth dimension is as described
            in `argmax`.
        """
        if argmax:
            return self.sess.run(self.predictions_argmax,
                                 feed_dict={self.image_input: images,
                                            self.keep_prob: 1.0})
        else:
            return self.sess.run(self.softmax_output,
                                 feed_dict={self.image_input: images,
                                            self.keep_prob: 1.0})

    def predict_and_save(self,
                         results_dir,
                         images_dir,
                         color_map,
                         resize=False,
                         image_file_extension='png',
                         include_unprocessed_image=False,
                         arrangement='vertical',
                         overwrite_existing=True):
        """
        Makes predictions for all images in a given directory, overlays a copy of the
        input images with the respective predictions, and saves the resulting images to disk.

        Arguments:
            results_dir (string): The directory in which to save the annotated prediction
                output images.
            images_dir (string): The directory in which the images to be processed are located.
            color_map (dictionary): A Python dictionary whose keys are non-negative
                integers representing segmentation classes and whose values are 1D tuples
                (or lists, Numpy arrays) of length 4 that represent the RGBA color values
                in which the respective classes are to be annotated. For example, if the
                dictionary contains the key-value pair `{1: (0, 255, 0, 127)}`, then
                this means that all pixels in the predicted image segmentation that belong
                to segmentation class 1 will be colored in green with 50% transparency
                in the input image.
            resize (tuple): `False` or a tuple of the form `(image_height, image_width)` that
                represents the size to which all images will be resized.
            image_file_extension (string, optional): The file extension of the
                images in the datasets. Must be identical for all images in all
                datasets in `datasets`. Defaults to `png`.
            include_unprocessed_image (bool, optional): If `True`, creates split view images
                containing both the input image and the overlayed segmented image.
                Defaults to `False`.
            arrangement (string, optional): Only relevant if `include_unprocessed_image` is `True`.
                Determines the arrangement for the split view. Can be either of 'vertical', meaning
                the processed and unprocessed images will be above each other, or 'horizontal',
                meaning the processed and unprocessed images will be next to each other.
                Defaults to 'vertical'.
            overwrite_existing (bool, optional): If `True`, overwrites the output directory
                in case it already exists.
        """

        # Make a directory in which to store the results.
        if overwrite_existing and os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)

        image_paths = glob(os.path.join(images_dir, '*.' + image_file_extension))
        num_images = len(image_paths)

        print('The segmented images will be saved to "{}"'.format(results_dir))

        tr = trange(num_images, file=sys.stdout)
        tr.set_description('Processing images')

        for i in tr:

            filepath = image_paths[i]

            image = scipy.misc.imread(filepath)
            if resize and not np.array_equal(image.shape[:2], resize):
                image = scipy.misc.imresize(image, resize)
            img_height, img_width, img_ch = image.shape

            prediction = self.predict([image], argmax=False)
            processed_image = np.asarray(
                print_segmentation_onto_image(image=image, prediction=prediction, color_map=color_map), dtype=np.uint8)

            if include_unprocessed_image:
                if arrangement == 'vertical':
                    output_width = img_width
                    output_height = 2 * img_height
                    processed_image = create_split_view(target_size=(output_height, output_width),
                                                        images=[processed_image, image],
                                                        positions=[(0, 0), (img_height, 0)],
                                                        sizes=[(img_height, img_width), (img_height, img_width)])
                else:
                    output_width = 2 * img_width
                    output_height = img_height
                    processed_image = create_split_view(target_size=(output_height, output_width),
                                                        images=[processed_image, image],
                                                        positions=[(0, 0), (0, img_width)],
                                                        sizes=[(img_height, img_width), (img_height, img_width)])

            scipy.misc.imsave(os.path.join(results_dir, os.path.basename(filepath)), processed_image)

    def save(self,
             model_save_dir,
             saver,
             tags=['default'],
             name=None,
             include_global_step=True,
             include_last_training_loss=True,
             include_metrics=True,
             force_save=False,
             freeze_model=False):
        """
        Saves the model to disk.

        Arguments:
            model_save_dir (string): The full path of the directory to which to
                save the model.
            saver (string, optional): Which saver to use when saving the model during training.
                Can be either of 'saved_model' in order to use `tf.saved_model` or 'train_saver'
                in order to use `tf.train.Saver`. Defaults to `tf.saved_model`. Check the
                TensorFlow documentation for details on which saver might be better for your
                use case. In general you can't go wrong with either of the two.
            tags (list, optional): An optional list of tags to save the model metagraph
                with in the SavedModel protocol buffer. Defaults to a list only containing
                the tag 'default'. At least one tag must be given.
            name (string, optional): An optional name that will be part of the name of the
                saved model's parent directory. Since you have the possibility to include
                the global step number and the values of metrics in the model name, giving
                an explicit name here is often not necessary.
            include_global_step (bool, optional): Whether or not to include the global
                step number in the model name. Defaults to `True`.
            include_last_training_loss (bool, optional): Whether of not to include the
                last training loss value in the model name. Defaults to `True`.
            include_metrics (bool, optional): If `True`, the last values of all recorded
                metrics will be included in the model name. Defaults to `True`.
            force_save (bool, optional): If `True`, force the saver to save the model
                even if no variables have changed since saving last. Defaults to `False`.
            freeze_model (bool, optional): If 'True', it will save a frozen model
        """

        if (not self.variables_updated) and (not force_save):
            print("Abort: Nothing to save, no training has been performed since the model was last saved.")
            return

        if not saver in {'saved_model', 'train_saver'}:
            raise ValueError(
                "Unexpected value for `saver`: Can be either 'saved_model' or 'train_saver', but received '{}'.".format(
                    saver))

        if self.training_loss is None:
            include_last_training_loss = False

        model_name = 'saved_model'
        if not name is None:
            model_name += '_' + name
        if include_global_step:
            self.g_step = self.sess.run(self.global_step)
            model_name += '_(globalstep-{})'.format(self.g_step)
        if include_last_training_loss:
            model_name += '_(trainloss-{:.4f})'.format(self.training_loss)
        if include_metrics:
            if self.eval_dataset == 'val':
                model_name += '_(eval_on_val_dataset)'
            else:
                model_name += '_(eval_on_train_dataset)'
            for i in range(len(self.metric_names)):
                try:
                    model_name += '_({}-{:.4f})'.format(self.metric_names[i], self.metric_values[i])
                except IndexError:
                    model_name += '_{}'.format(time.time())
        if not (include_global_step or include_last_training_loss or include_metrics) and (name is None):
            model_name += '_{}'.format(time.time())

        if saver == 'saved_model':
            saved_model_builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(model_save_dir, model_name))
            saved_model_builder.add_meta_graph_and_variables(sess=self.sess, tags=tags)
            saved_model_builder.save()
        else:
            saver = tf.train.Saver(var_list=None,
                                   reshape=False,
                                   max_to_keep=5,
                                   keep_checkpoint_every_n_hours=1.0)
            if not os.path.exists(os.path.join(model_save_dir, model_name)):
                os.makedirs(os.path.join(model_save_dir, model_name))
            saver.save(self.sess,
                       save_path=os.path.join(model_save_dir, model_name, 'saved_model'),
                       write_meta_graph=True,
                       write_state=True)

        self.variables_updated = False

        if freeze_model:
            from tensorflow.python.tools import freeze_graph
            tf.train.write_graph(self.sess.graph.as_graph_def(), os.path.join(model_save_dir, model_name),
                                 'frozen_model.pb', as_text=False)

    def load_variables(self, path):
        '''
        Load variable values into the current model. Only works for variables that
        were saved with 'train_saver'. See `save()` for details.
        '''
        saver = tf.train.Saver(var_list=None)
        saver.restore(self.sess, path)

    def close(self):
        '''
        Closes the session. This method is important to call when you are done working
        with the model in order to release the resources it occupies.
        '''
        self.sess.close()
        print("The session has been closed.")


if __name__ == "__main__":
    config = networkConfig()
    _model = Inceptionv1FCN8s(experiment_dir="./", exp_config=config)
    saver = tf.train.Saver()
    _data = np.ones(shape=(config.batch_size, config.image_height,
                           config.image_width, config.image_depth), dtype=np.float32)
    _labels = np.ones(shape=(config.batch_size, config.image_height,
                             config.image_width, config.n_classes), dtype=np.float32)
    keep_prob = 0.5

    with _model.sess as sess:
        logits = sess.run(_model.fcn8s_output, feed_dict={_model.image_input: _data, _model.labels: _labels,
                                                          _model.keep_prob: keep_prob})
        # Save the variables to disk.
        save_path = saver.save(sess, "./test_output/model.ckpt")
        print("Model saved in path: %s" % save_path)

        model_vars = tf.trainable_variables()

        for ivar in model_vars:
            print(ivar)

    print("Test Passed")
