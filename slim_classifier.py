import tensorflow as tf
import tensorflow.contrib.slim as slim

import load_data as ld

# define an 11-level NN comprising CNN and fully connected layers
def build_classifier(inputs):

    #setup layer variables
    with slim.arg_scope([slim.conv2d],
                        kernel_size=[3, 3],
                        padding='SAME',
                        biases_initializer=tf.constant_initializer(0.0)):
        with slim.arg_scope([slim.max_pool2d],
                            kernel_size=[2, 2],
                            padding='SAME'):

            #flatten input image
            net = slim.flatten(inputs)
            # conv1
            net = slim.repeat(net, 2, slim.conv2d, 64, scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')
            # conv2
            net = slim.repeat(net, 2, slim.conv2d, 128, scope='conv2')
            net = slim.max_pool2d(net, scope='pool2')

            # reshape tensor to matrix
            net = slim.flatten(net)
            # fc3
            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.dropout(net, 0.5, scope='dropout3')
            # fc4
            net = slim.fully_connected(net, 256, scope='fc4')
            net = slim.dropout(net, 0.5, scope='dropout4')
            # linear prediction, override the activation_fn in scope
            net = slim.fully_connected(net, 10, activation_fn=None, scope='linear')
            # softmax4
            # categorical probability distribution over output vector
            outputs = slim.softmax(net, scope='softmax4')
            return outputs

#set up the graph
g = tf.Graph()
with g.as_default():
    # 4D Tensor placeholder for input images
    inputs = tf.placeholder(tf.float32, shape=[None] + [256, 256, 3], name="images")

    with tf.variable_scope("TF-Slim", [inputs]):
        # add model to graph
        outputs = build_classifier(inputs)

