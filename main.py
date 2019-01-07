import tensorflow as tf
import numpy as np
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # shut tensorflow up

#the home-made stuff
import load_data as ld
import slim_classifier as sc
import slim_autoencoder as ae

#perform t_sne analysis on lsr representation of the image
def t_sne(label_stack, lsr_stack):

    feat_cols = ['dimension' + str(i) for i in range(lsr_stack.shape[1])]
    df = pd.DataFrame(lsr_stack, columns=feat_cols)
    #print(lsr_stack.shape[1])
    #print(label_stack.shape)
    #print(label_stack[0][0].shape)
    #df['label'] = label_stack[0]    # need to process labels noise/not to make sense

    #TODO restart here. You're fixing the labels to be obvisou nosis/non-noise. currently working on slim_autoencoder right at the bottom
    # future me belives in you


print("Main executing")

#ld.create_dataset(write_addr= "apmldataset_denoised.tfrecords", all = False)
#ld.create_dataset(write_addr= "apmldataset.tfrecords", all = True)
#ld.create_evaluation_dataset(write_addr = "apmldataset_evaluation.tfrecords")

#ae.train_autoencoder()

label_stack, lsr_stack = ae.encode(100)
t_sne(label_stack, lsr_stack)

#sc.train_classifier()
#sc.evalute_classifier()
#sc.test_classifier()

print("Main complete")


