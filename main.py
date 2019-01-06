import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # shut tensorflow up

#the home-made stuff
import load_data as ld
import slim_classifier as sc
import slim_autoencoder as ae


print("Main executing")

#ld.create_dataset(write_addr= "apmldataset_denoised.tfrecords", all = False)
#ld.create_dataset(write_addr= "apmldataset.tfrecords", all = True)
#ld.create_evaluation_dataset(write_addr = "apmldataset_evaluation.tfrecords")

#ae.train_autoencoder()

#sc.train_classifier()
sc.evalute_classifier()

print("Main complete")


