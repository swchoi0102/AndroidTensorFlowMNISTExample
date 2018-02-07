import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def separable_conv2d(x, W1, W2, strides=1):
    x = tf.nn.separable_conv2d(x, W1, W2, strides=[1, strides, strides, 1], padding='SAME')
    return tf.nn.relu(x)


def masked_conv2d(x, W, b, strides=1, name=None):
    with tf.variable_scope(name) as scope:
        x = tf.nn.conv2d(x, pruning.apply_mask(W), strides=[1, strides, strides, 1], padding='SAME', name=scope.name)
        x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

