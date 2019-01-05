import tensorflow as tf
import numpy as np


import load_data as ld

#load from the tfrecords file
# TODO add option to convert hair colour encoding to one-hot
def parser(record):
    keys_to_features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([5], tf.int64)
    }
    parsed = tf.parse_single_example(record, features=keys_to_features)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed['label'], tf.int32)

    return {'image': image}, label

def import_dataset(addr):
    dataset = tf.data.TFRecordDataset(filenames=addr, num_parallel_reads=4) # check npr if this is weird
    dataset = dataset.map(parser)
    return dataset


print("Main executing")

#ld.create_dataset()

#for example in tf.python_io.tf_record_iterator("apmldataset.tfrecords"):
 # result = tf.train.Example.FromString(example)
 # print(result.features.feature['label'].int64_list.value)
  #print(result.features.feature['image'].bytes_list.value)

raw_dataset = import_dataset("apmldataset.tfrecords")

ae.autoencoder(raw_dataset)



print("Main complete")


