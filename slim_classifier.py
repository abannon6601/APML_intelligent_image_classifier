import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys
import csv

import load_data as ld

sys.path.append("../")

#checkpoint saving
save_path = './saves_classifier/classifier.ckpt'

#training params
learning_rate = 0.01
num_epochs = 20

#data params
total_images = 4000 #80%, used in training
batch_size = 10

#model params
n_nodes_inpl = [256, 256, 3]

# define NN comprising convolusional and fully connected layers
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
            net = slim.repeat(inputs, 2, slim.conv2d, 4, scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')
            # conv2s
            net = slim.repeat(net, 2, slim.conv2d, 6, scope='conv2')
            net = slim.max_pool2d(net, scope='pool2')

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

# train the classifier
def train_classifier():
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

            # save object
            saver = tf.train.Saver(tf.all_variables())

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
                    #save the model
                    saved_path = saver.save(sess, save_path)
                    print("Saved in path: %s" % saved_path)

            except Exception as e:
                #we hit a mine, so stop doing shit
                coord.request_stop()
                print(e)
                coord.request_stop(e)
            finally:
                #Shut it down! Code red! Burn the evidence and run!
                coord.request_stop()
                coord.join(threads)

def test_classifier():
    print("Reloading model for test")

    # number of images to test on
    test_num = 1000 #20%

    #holding arrays for results of test
    hair_preds = np.zeros(test_num)
    hair_trues = np.zeros(test_num)
    glasses_preds = np.zeros(test_num)
    glasses_trues = np.zeros(test_num)
    smile_preds = np.zeros(test_num)
    smile_trues = np.zeros(test_num)
    young_preds = np.zeros(test_num)
    young_trues = np.zeros(test_num)
    human_preds = np.zeros(test_num)
    human_trues = np.zeros(test_num)

    g = tf.Graph()
    with g.as_default():
        # in-out placeholders
        inputs = tf.placeholder(tf.float32, shape=[None] + [256, 256, 3])
        labels = tf.placeholder('float', shape=[None] + [10])

        with tf.variable_scope("TF-Slim", [inputs]):
            # add model to graph
            outputs = build_classifier(inputs)

        restorer = tf.train.Saver()

        with tf.Session() as sess:
            restorer.restore(sess, save_path)
            print('model restored')
            filename = './apmldataset_denoised.tfrecords'
            filename_queue = tf.train.string_input_producer([filename])
            image, label = ld.read_and_decode_no_label_mod(filename_queue, n_nodes_inpl)    # use the no label mod option to get the orgignal hiar colour encoding
            image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=1, capacity=100,
                                                              num_threads=1,
                                                              min_after_dequeue=0)  # set the batch size to one to simplify processing

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)


            for test in range(test_num):  # repeat for all images
                # get new data
                input_im, input_lab = sess.run([image_batch, label_batch])

                # process the model and split into hair/other classses
                result_ten = sess.run(outputs, feed_dict={inputs: input_im})
                hair_col, eyeg_smil_yon_hu, = result_ten[:, 0:5], result_ten[:, 5:]

                # convert back to original label form
                bin_vec = np.rint(sess.run(tf.round(tf.math.sigmoid(eyeg_smil_yon_hu)))[0])
                hair_final = np.argmax(sess.run(tf.nn.softmax(hair_col)))

                # write to the reslts arrays
                hair_preds[test] = hair_final
                hair_trues[test] = input_lab[0][0]

                glasses_preds[test] = bin_vec[0]
                glasses_trues[test] = input_lab[0][1]

                smile_preds[test] = bin_vec[1]
                smile_trues[test] = input_lab[0][2]

                young_preds[test] = bin_vec[2]
                young_trues[test] = input_lab[0][3]

                human_preds[test] = bin_vec[3]
                human_trues[test] = input_lab[0][4]


            # write the confusion matrices
            hair_cm = sess.run(tf.contrib.metrics.confusion_matrix(hair_trues, hair_preds, num_classes = 6))
            glasses_cm = sess.run(tf.contrib.metrics.confusion_matrix(glasses_trues, glasses_preds))
            smile_cm = sess.run(tf.contrib.metrics.confusion_matrix(smile_trues, smile_preds))
            young_cm = sess.run(tf.contrib.metrics.confusion_matrix(young_trues, young_preds))
            human_cm = sess.run(tf.contrib.metrics.confusion_matrix(human_trues, human_preds))

            #normalise
            hair_cm = np.true_divide(hair_cm,np.sum(hair_cm))
            glasses_cm = np.true_divide(glasses_cm, np.sum(glasses_cm))
            smile_cm = np.true_divide(smile_cm, np.sum(smile_cm))
            young_cm = np.true_divide(young_cm, np.sum(young_cm))
            human_cm = np.true_divide(human_cm, np.sum(human_cm))

            print("hair")
            print(hair_cm)
            print("glasses")
            print(glasses_cm)
            print("smile")
            print(smile_cm)
            print("young")
            print(young_cm)
            print("human")
            print(human_cm)



#load the saved model for evaluation
def evalute_classifier():
    print("Reloading model for evaluation")

    # how many images there are to evaluate
    image_num = 100

    open('./csv_files/task_1.csv', mode='w')# clear the files or create them if they don't already exist
    open('./csv_files/task_2.csv', mode='w')
    open('./csv_files/task_3.csv', mode='w')
    open('./csv_files/task_4.csv', mode='w')
    open('./csv_files/task_5.csv', mode='w')


    g = tf.Graph()
    with g.as_default():
        # in-out placeholders
        inputs = tf.placeholder(tf.float32, shape=[None] + [256, 256, 3])
        labels = tf.placeholder('float', shape=[None] + [10])

        with tf.variable_scope("TF-Slim", [inputs]):
            # add model to graph
            outputs = build_classifier(inputs)

        restorer = tf.train.Saver()

        with tf.Session() as sess:
            restorer.restore(sess, save_path)
            print('model restored')
            filename = './apmldataset_evaluation.tfrecords'
            filename_queue = tf.train.string_input_producer([filename])
            image, label = ld.read_and_decode_evaluation(filename_queue, n_nodes_inpl)
            image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=1, capacity=100,
                                                              num_threads=1, min_after_dequeue=0)   # set the batch size to one to simplify processing

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)


            for i in range(image_num): #repeat for all images
                #get new data
                input_im, input_lab = sess.run([image_batch, label_batch])

                # process the model and split into hair/other classses
                result_ten = sess.run(outputs, feed_dict={inputs: input_im})
                hair_col, eyeg_smil_yon_hu, = result_ten[:, 0:5], result_ten[:, 5:]

                #convert back to original label form
                bin_vec = sess.run(tf.round(tf.math.sigmoid(eyeg_smil_yon_hu)))[0]
                bin_vec = (bin_vec*2)-1 # convert back to 1/-1 encoding
                hair_final = np.argmax(sess.run(tf.nn.softmax(hair_col)))

                #one-hot encoded section
                hair_classes = tf.one_hot(hair_final, 6)

                #extrat the image name
                image_id = sess.run(label_batch)[0][0]
                #print("image " + str(image_id))
                #print(bin_vec)
                #print(hair_final)

                #write to the files
                with open('./csv_files/task_1.csv', mode='a', newline='') as task_1_file:
                    employee_writer = csv.writer(task_1_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    employee_writer.writerow([str(image_id)+ "", str(int(bin_vec[1]))])

                with open('./csv_files/task_2.csv', mode='a', newline='') as task_1_file:
                    employee_writer = csv.writer(task_1_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    employee_writer.writerow([str(image_id) + "", str(int(bin_vec[2]))])

                with open('./csv_files/task_3.csv', mode='a', newline='') as task_1_file:
                    employee_writer = csv.writer(task_1_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    employee_writer.writerow([str(image_id)+ "", str(int(bin_vec[0]))])

                with open('./csv_files/task_4.csv', mode='a', newline='') as task_1_file:
                    employee_writer = csv.writer(task_1_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    employee_writer.writerow([str(image_id)+ "", str(int(bin_vec[3]))])

                with open('./csv_files/task_5.csv', mode='a', newline='') as task_1_file:
                    employee_writer = csv.writer(task_1_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    employee_writer.writerow([str(image_id)+ "", str(hair_final)])


