import tensorflow as tf
import tensorflow.contrib.slim as slim

import load_data as ld

import sys
sys.path.append("../")

n_nodes_inpl = [256, 256, 3]

learning_rate = 0.00001
num_epochs = 30

def inspect_variables(variables):
    for var in variables:
            print('name = {} {}shape = {}'.format(var.name, " "*(55-len(var.name)), var.get_shape()))
    print()

def inspect_layers(endpoints):
    for k, v in endpoints.iteritems():
        print('name = {} {}shape = {}'.format(v.name, " "*(55-len(v.name)), v.get_shape()))
    print()

def build_encoder(inputs):

    with slim.arg_scope([slim.fully_connected],
                        kernel_size=[3, 3],
                        padding='SAME',
                        biases_initializer=tf.constant_initializer(0.0)):

    net = slim.flatten(inputs)

    net = slim.fully_connected(net, 128,
                                    weights_initializer = tf.random_normal_initializer(
                                    stddev=0.1),
                                    scope='fc1')
    lsr = slim.fully_connected(net, 32,
                                    weights_initializer=tf.random_normal_initializer(
                                    stddev=0.1),
                                    scope='fc2')

    return lsr  # this should now contain the latent space representation

def build_decoder(lsr):

    net = slim.fully_connected(lsr, 32,
                                    weights_initializer = tf.random_normal_initializer(
                                    stddev=0.1),
                                    scope='fc3')
    net = slim.fully_connected(net, 256*256*3,
                                    weights_initializer=tf.random_normal_initializer(
                                    stddev=0.1),
                                    scope='fc4')

    net = tf.reshape(net, [-1, 256, 256, 3])

    return net

print("Slim-autoencoder running")

#construct the autoencoder graph
g = tf.Graph()
with g.as_default():

    # 4D Tensor placeholder for input images
    inputs = tf.placeholder(tf.float32, shape=[None] + [256, 256, 3], name="images")

    with tf.variable_scope("TF-Slim", [inputs]):
        # add model to graph
        lsr = build_encoder(inputs)
        result = build_decoder(lsr)

    #define optimiser and loss function
    loss = tf.reduce_sum(tf.square(result-inputs))
    optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Initialize the variables
    init = tf.group(tf.initialize_local_variables(), tf.global_variables_initializer())

    print("autoencoder starting tf.session")


    with tf.Session() as sess:

        sess.run(init)

        filename = './apmldataset.tfrecords'
        filename_queue = tf.train.string_input_producer([filename])
        image, label = ld.read_and_decode(filename_queue,n_nodes_inpl)
        images, labels = tf.train.shuffle_batch([image, label], batch_size= 10, capacity= 80, num_threads=1, min_after_dequeue=5)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            for epoch in range(num_epochs):

                input_x,_ = sess.run([images, labels])
                _, l1 = sess.run([optimiser,loss], feed_dict = {inputs: input_x})
                print('loss ' + str(l1))

        except Exception as e:
            #we hit a mine, so stop doing shit
            coord.request_stop()
        finally:
            #Shut it down! Code red! burn the evidence and run!
            coord.request_stop()
            coord.join(threads)


12