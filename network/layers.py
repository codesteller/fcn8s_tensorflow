import tensorflow as tf
import numpy as np


def conv2d(self, layer_name, inputs, out_channels, kernel_size, strides=1, padding='SAME'):
    in_channels = inputs.get_shape()[-1]
    with tf.variable_scope(layer_name) as scope:
        self.scope[layer_name] = scope
        w = tf.get_variable(name='weights',
                            trainable=True,
                            shape=[kernel_size, kernel_size, in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases',
                            trainable=True,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        inputs = tf.nn.conv2d(inputs, w, [1, strides, strides, 1], padding=padding, name='conv')
        inputs = tf.nn.bias_add(inputs, b, name='bias_add')
        inputs = tf.nn.relu(inputs, name='relu')
        return inputs
    
def max_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.max_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)

def avg_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
    with tf.name_scope(layer_name):
        return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                name=layer_name)

def lrn(self, layer_name, inputs, depth_radius=5, alpha=0.0001, beta=0.75):
    with tf.name_scope(layer_name):
        return tf.nn.local_response_normalization(name='pool1_norm1', input=inputs, depth_radius=depth_radius,
                                                    alpha=alpha, beta=beta)

def concat(self, layer_name, inputs):
        with tf.name_scope(layer_name):
            one_by_one = inputs[0]
            three_by_three = inputs[1]
            five_by_five = inputs[2]
            pooling = inputs[3]
            return tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)

def dropout(self, layer_name, inputs, keep_prob):
    # dropout_rate = 1 - keep_prob
    with tf.name_scope(layer_name):
        return tf.nn.dropout(name=layer_name, x=inputs, keep_prob=keep_prob)

def bn(self, layer_name, inputs, epsilon=1e-3):
    with tf.name_scope(layer_name):
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        inputs = tf.nn.batch_normalization(inputs, mean=batch_mean, variance=batch_var, offset=None,
                                            scale=None, variance_epsilon=epsilon)
        return inputs

class Config:
    def __init_(self):
        self.batch_size = 1
        self.image_width = 2048
        self.image_height = 1024
        self.image_depth = 3
        self.n_classes = 34
