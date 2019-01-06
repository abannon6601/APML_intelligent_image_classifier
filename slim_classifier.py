import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys

import load_data as ld

learning_rate = 0.01
num_epochs = 10

total_images = 5000
batch_size = 10

n_nodes_inpl = [256, 256, 3]

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

            # conv1
            net = slim.repeat(inputs, 1, slim.conv2d, 4, scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')
            # conv2
            #net = slim.repeat(net, 2, slim.conv2d, 32, scope='conv2')
            #net = slim.max_pool2d(net, scope='pool2')

            # reshape tensor to matrix
            net = slim.flatten(net)
            # fc3
            #net = slim.fully_connected(net, 1024, scope='fc3')
            #net = slim.dropout(net, 0.5, scope='dropout3')
            # fc4
            net = slim.fully_connected(net, 32, scope='fc4')
            net = slim.dropout(net, 0.5, scope='dropout4')
            # linear prediction, override the activation_fn in scope
            outputs = slim.fully_connected(net, 10, activation_fn=None, scope='linear') # 10 output classes
            # softmax4
            # categorical probability distribution over output vector
            #outputs = slim.softmax(net, scope='softmax4')

            return outputs

#set up the graph
g = tf.Graph()
with g.as_default():
    # in-out placeholders
    inputs = tf.placeholder(tf.float32, shape=[None] + [256, 256, 3])
    labels = tf.placeholder('float', shape=[None] + [10])

    with tf.variable_scope("TF-Slim", [inputs]):
        # add model to graph
        outputs = build_classifier(inputs)

    #define optimiser and loss function
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs)
    optimiser = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    # Initialize the variables
    init = tf.group(tf.initialize_local_variables(), tf.global_variables_initializer())

    print("Classifier beginning tf session for training")

    with tf.Session() as sess:

        sess.run(init)

        filename = './apmldataset_denoised.tfrecords'
        filename_queue = tf.train.string_input_producer([filename])
        image, label = ld.read_and_decode(filename_queue,n_nodes_inpl)
        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size= batch_size, capacity= 80, num_threads=1, min_after_dequeue=5)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch in range(int(total_images/batch_size)):
                    input_im, input_lab = sess.run([image_batch, label_batch])
                    #print(input_lab.shape)
                    #print(input_lab)
                    _, l1 = sess.run([optimiser,loss], feed_dict={inputs: input_im, labels: input_lab})
                    epoch_loss += np.sum(l1)
                    #print("batch loss " + str(np.sum(l1)), flush = True)
                print("epoch loss " + str(epoch_loss))

        except Exception as e:
            #we hit a mine, so stop doing shit
            coord.request_stop()
            print(e)
            coord.request_stop(e)
        finally:
            #Shut it down! Code red! Burn the evidence and run!
            coord.request_stop()
            coord.join(threads)


