#OLD AND NO LONGER USED



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import load_data as ld

#NOTE images are 256 x 256

def autoencoder(raw_dataset):

    print("autoencoder is in startup")

    # Training Parameters
    learning_rate = 0.01
    num_steps = 30000

    display_step = 3000

    # Network Parameters
    num_hidden_1 = 64 # 1st layer num features
    num_hidden_2 = 32 # 2nd layer num features                 DEFINES FEATURE SPACE
    num_input = [256,256,3] #

    # tf Graph input (only pictures)
    input_X = tf.placeholder("float", [None] + num_input)

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([num_input])),
    }

    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        return layer_2

    # Building the decoder
    def decoder(x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
        return layer_2

    # Construct model
    encoder_op = encoder(input_X)
    decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # Targets are the input data.
    y_true = input_X

    # Define loss and optimizer, minimize the squared error
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    # Initialize the variables
    init = tf.group(tf.initialize_local_variables(), tf.global_variables_initializer())

    #split the dataset into training, test, and add a batch size. Note: we don't need
    #an evaluation set becuase the autoencoder is unsupervised

    train_dataset = raw_dataset.take(4000)  # 4/5 of the data
    test_dataset =  raw_dataset.skip(4000)  # 1/5 of the data
    BATCH_SIZE = 5 # to keep things simple, make this a factor of both datasets

    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)



    #start training with a new session
    print("autoencoder starting tf.session")
    with tf.Session() as sess:

        #train
        for i in range(1, num_steps + 1):

            #image_batch, _ = train_iterator.get_next() # we only care about the images so ignore the labels
            #print(image_batch)
            #image_batch, label_batch = tf.contrib.data.get_single_element(train_dataset)
            #image_t = sess.run(image_test)
            #print(result)
            #print(image_t)



            filename = './apmldataset.tfrecords'
            filename_queue = tf.train.string_input_producer([filename])

            image , _ = ld.read_and_decode(filename_queue, num_input)

            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)

            image_proc = sess.run(image)

            print(image_proc)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={input_X: image_proc})
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))

    print("autoencoder complete")

    return 0

