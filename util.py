import pickle
import numpy as np
import tensorflow as tf
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def data_read(train_file):
    x = np.array([])
    label = None
    for file in train_file:
        dict = unpickle(file)
        # if x == None:
        #     x = dict[b'data']
        #     label = dict[b'labels']
        # else:
        x = np.concatenate((x,dict[b'data'].flatten()))
        nums = tf.one_hot(dict[b'labels'],depth=10)
        if(label==None):
            label = nums
        else:
            tf.concat([label,nums],axis=0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        label = sess.run(label)
    return x, label
