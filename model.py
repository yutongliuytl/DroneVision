import tensorflow as tf 
import numpy as np 


# Convolution wrapper with bias and relu activation
def conv(x, weights, biases, strides=1):
    x = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, biases)
    return tf.nn.relu(x) 

# Max pooling wrapper
def maxpool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv_network(x, weights, biases):  
    # Input layer (ouputs a 64 x 64)
    conv1 = conv(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool(conv1)

    # Convolution Layer (outputs a 32 x 32 and then a 16 x 16)
    conv2 = conv(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool(conv2)
    conv3 = conv(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool(conv3)

    # Dense layer (Flattening + Fully Connected layers)
    dl = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    dl = tf.add(tf.matmul(dl, weights['wd1']), biases['bd1'])
    dl = tf.nn.relu(dl)

    # Output layer
    output = tf.add(tf.matmul(dl, weights['out']), biases['out'])
    return output

