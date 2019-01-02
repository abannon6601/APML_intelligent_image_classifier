# this is a one-use set of functions to read the main dataset into a tfrecords file
# during normal operations this file is NEVER USED

global_address = 'C:\\Users\\Alan\\Desktop\\OneDrive - University College London\\Year 4\\Machine learning\\assignment_data'
# imgaes are 256x256

import os
import tensorflow as tf
import pandas
import numpy as np
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
def load_data(addr):
    print("Loading Data: START")

    labels = pandas.read_csv(os.path.join(addr, 'attribute_list.csv')).values;

    with tf.python_io.TFRecordWriter("apmldataset.tfrecords") as writer:
        firstLine = True
        pbar = tqdm(labels)
        image_number = 0
        for row in pbar:
            if firstLine: # ignore the header of the csv file
                firstLine = False
                continue

            image_number = image_number + 1
            pbar.set_description("Loading image %d" % image_number)

            try:
                path = os.path.join(addr, 'dataset', row[0] + '.png')
                image = load_image(path)

                if image is not None:
                    flat_image = cv2.imencode('.jpg', image)[1].tostring()

                    feature = {
                        'label': _int64_feature([int(i) for i in row[1:]]),
                        'image': _bytes_feature(tf.compat.as_bytes(flat_image))
                    }
                    tf_example = tf.train.Example(features = tf.train.Features(feature = feature))
                    writer.write(tf_example.SerializeToString())

                else:
                    print('IMAGE NOT FOUND')
            except Exception as inst:
                print(inst)
                pass

        writer.close()


        print("Loading Data: DONE")

#quickly print out all the labels of the dataset for debug
def view_data_labels():
    for example in tf.python_io.tf_record_iterator("apmldataset.tfrecords"):
        result = tf.train.Example.FromString(example)
        print(result.features.feature['train/label'].int64_list.value)

#quick function to run with local path
def create_dataset():
    load_data(global_address)