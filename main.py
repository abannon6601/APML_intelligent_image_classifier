import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # shut tensorflow up

#the home-made stuff
import load_data as ld
import slim_classifier as sc
import slim_autoencoder as ae
from ggplot import *

#perform t_sne analysis on lsr representation of the image
def t_sne(label_stack, lsr_stack):

    feat_cols = ['dimension' + str(i) for i in range(lsr_stack.shape[1])]
    df = pd.DataFrame(lsr_stack, columns=feat_cols)
    df['label'] = label_stack    # need to process labels noise/not to make sense

    #create a further random vector
    rndperm = np.random.permutation(df.shape[0])
    print('Size of the dataframe: {}'.format(df.shape))

    n_sne = df.shape[0]

    tsne = TSNE(n_components=2, verbose=1, perplexity=80, n_iter=300)
    tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

    df_tsne = df.loc[rndperm[:n_sne],:].copy()
    df_tsne['x-tsne'] = tsne_results[:,0]
    df_tsne['y-tsne'] = tsne_results[:,1]


    chart = ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) \
            + geom_point(size=70, alpha=0.3) \
            + ggtitle("tSNE dimensions colored on noise/non-noise")
    print(chart)



print("Main executing")

#ld.create_dataset(write_addr= "apmldataset_denoised.tfrecords", all = False)
#ld.create_dataset(write_addr= "apmldataset.tfrecords", all = True)
#ld.create_evaluation_dataset(write_addr = "apmldataset_evaluation.tfrecords")

#ae.train_autoencoder()

#label_stack, lsr_stack = ae.encode(5000)
#t_sne(label_stack, lsr_stack)

#sc.train_classifier()
#sc.evalute_classifier()
sc.test_classifier()

print("Main complete")


