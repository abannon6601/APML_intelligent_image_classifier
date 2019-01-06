import tensorflow as tf
import tensorflow.contrib.slim as slim

import load_data as ld

import sys
sys.path.append("../")

save_path = './saves_autoencoder/autoencoder.ckpt'

#model params
n_nodes_inpl = [256, 256, 3]

#training params
learning_rate = 0.01
num_epochs = 3

#data params
total_images = 5000
batch_size = 100


def build_encoder(inputs):

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


def train_autoencoder():

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
        loss = tf.reduce_mean(tf.square(result-inputs))
        optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Initialize the variables
        init = tf.group(tf.initialize_local_variables(), tf.global_variables_initializer())

        print("autoencoder starting tf.session")


        with tf.Session() as sess:

            sess.run(init)

            # save object
            saver = tf.train.Saver(tf.all_variables())

            filename = './apmldataset.tfrecords'
            filename_queue = tf.train.string_input_producer([filename])
            image, label = ld.read_and_decode(filename_queue,n_nodes_inpl)
            images, labels = tf.train.shuffle_batch([image, label], batch_size= batch_size, capacity= 800, num_threads=1, min_after_dequeue=50)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                for epoch in range(num_epochs):
                    epoch_loss = 0
                    for batch in range(int(total_images/batch_size)):
                        input_x,_ = sess.run([images, labels])
                        _, l1 = sess.run([optimiser, loss], feed_dict={inputs: input_x})
                        epoch_loss += l1
                        print("batch loss " + str(l1))
                    print('Epoch loss ' + str(epoch_loss))
                    #save the model
                    saved_path = saver.save(sess, save_path)
                    print("Saved in path: %s" % saved_path)



            except Exception as e:
                #we hit a mine, so stop doing shit
                print(e)
                coord.request_stop(e)
            finally:
                #Shut it down! Code red! Burn the evidence and run!
                coord.request_stop()
                coord.join(threads)


