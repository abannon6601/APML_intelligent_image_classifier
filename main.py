import tensorflow as tf


#load from the tfrecords file
# TODO add option to convert hair colour encoding to one-hot
def parser(record):
    keys_to_features = {
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image"], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed["label"], tf.int32)

    return {'image': image}, label


def import_dataset(addr):
    dataset = tf.data.TFRecordDataset(filenames=addr, num_parallel_reads=50) # check npr if this is weird
    dataset = dataset.map(parser)
    return dataset


print("Loading tfrecords file")

import_dataset("apmldataset.tfrecords")

print("done loading tfr file")

