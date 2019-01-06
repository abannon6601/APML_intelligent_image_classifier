import tensorflow as tf
import numpy as np


import load_data as ld


print("Main executing")

#ld.create_dataset(write_addr= "apmldataset_denoised.tfrecords", all = False)

ld.create_dataset(write_addr= "apmldataset.tfrecords", all = True)

#for example in tf.python_io.tf_record_iterator("apmldataset_denoised.tfrecords"):
#  result = tf.train.Example.FromString(example)
#  print(result.features.feature['label'].int64_list.value)


print("Main complete")


