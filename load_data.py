# this is a one-use set of functions to read the main dataset into a tfrecords file
# during normal operations this file is NEVER USED

global_address = 'C:\\Users\\Alan\\Desktop\\OneDrive - University College London\\Year 4\\Machine learning\\assignment_data'
# images are 256x256

import os
import tensorflow as tf
import pandas
import numpy as np
from sklearn.preprocessing import normalize
import cv2
from tqdm import tqdm

#wrapper functions to convert to feature
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#load a single image with cv2 and correct the colour to RGB (form BGR)
def load_image(addr):
    image = cv2.imread(addr)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # could resize here, but not needed
        return image.astype(np.float32)
    return image

#Create a tfrecords file with the entire dataset
def load_data(addr, out_addr = "apmldataset.tfrecords", all = True):
    print("Loading Data: START")

    labels = pandas.read_csv(os.path.join(addr, 'attribute_list.csv')).values;

    with tf.python_io.TFRecordWriter(out_addr) as writer:
        firstLine = True
        pbar = tqdm(labels)
        image_number = 0
        for row in pbar:
            if firstLine: # ignore the header of the csv file
                firstLine = False
                continue

            image_number = image_number + 1
            pbar.set_description("Loading image %d" % image_number)

            # create a noiseless tfrecords if flags correct             USE FOR TESTING ONLY!!! TODO disable before final tests
            if row[1] == "-1" and all == False:
                #print("skipping noise")
                continue

            try:
                path = os.path.join(addr, 'dataset', row[0] + '.png')
                image = load_image(path)


                if image is not None:
                    flat_image = cv2.imencode('.jpg', image)[1].tostring()

                    # apply normalisation to the classes to bring them to binary state
                    hcolour = row[1]    # preserve the hair colour
                    row = (row == "1").astype(int)
                    row[1] = hcolour

                    feature = {
                        'label': _int64_feature([int(i) for i in row[1:]]),
                        'image': _bytes_feature(tf.compat.as_bytes(flat_image))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

                else:
                    print('IMAGE NOT FOUND')
            except Exception as inst:
                print(inst)
                pass

        writer.close()
        print("Loading Data: DONE")

#Create a tfrecords file with the evaluation dataset
def load_data_evaluation(addr, out_addr = "apmldataset_evaluation.tfrecords", all = True):
    print("Loading Data: START")
    num_images = 100
    itr = range(num_images)

    with tf.python_io.TFRecordWriter(out_addr) as writer:
        pbar = tqdm(itr)
        image_number = 0
        for image in pbar:

            image_number = image_number + 1
            pbar.set_description("Loading image %d" % image_number)

            try:
                path = os.path.join(addr, 'testing_dataset', str(image_number) + '.png')
                image = load_image(path)

                if image is not None:
                    flat_image = cv2.imencode('.jpg', image)[1].tostring()

                    feature = {
                        'label': _int64_feature([image_number]),
                        'image': _bytes_feature(tf.compat.as_bytes(flat_image))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

                else:
                    print('IMAGE NOT FOUND')
            except Exception as inst:
                print("caught exception")
                print(inst)
                pass

        writer.close()
        print("Loading Data: DONE")


#quickly print out all the labels of the dataset for debug
def view_data_labels():
    for example in tf.python_io.tf_record_iterator("apmldataset.tfrecords"):
        result = tf.train.Example.FromString(example)
        print(result.features.feature['train/label'].int64_list.value)

#quick functions to run with local path
def create_dataset(write_addr, all = True):
    load_data(global_address, write_addr, all)
def create_evaluation_dataset(write_addr, all = True):
    load_data_evaluation(global_address, write_addr, all)

#read back the data from the tfrecords file
def read_and_decode(filename_queue,n_nodes_inpl):
    reader = tf.TFRecordReader()
    _, serialised_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialised_example, features={
        'label': tf.FixedLenFeature([5], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)})

    #grab, normalise, and reshape the image
    image = tf.image.decode_jpeg(features['image'])
    image /= 255
    image = tf.reshape(image, n_nodes_inpl)

    # grab label, multi-class label to binary classes and attach to the label
    label = features['label']
    hcolour = label[0]
    hair_classes = tf.one_hot(hcolour,6)    #six classes
    label = tf.concat([hair_classes,tf.to_float(label[1:])],axis=0) # concat disguards the old haircolour

    return image, label

#read back the data from the evaluation tfrecords file
def read_and_decode_evaluation(filename_queue,n_nodes_inpl):
    reader = tf.TFRecordReader()
    _, serialised_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialised_example, features={
        'label': tf.FixedLenFeature([1], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)})

    #grab, normalise, and reshape the image
    image = tf.image.decode_jpeg(features['image'])
    image /= 255
    image = tf.reshape(image, n_nodes_inpl)

    # grab label
    label = features['label']

    return image, label