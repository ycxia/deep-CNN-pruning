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
        x = np.concatenate((x,dict[b'data'].flatten()))
        nums = tf.one_hot(dict[b'labels'],depth=10)
        if(label==None):
            label = nums
        else:
            label = tf.concat([label,nums],axis=0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        label = sess.run(label)

    x = x/255.0
    x = np.reshape(x, (-1, 3, 32, 32))
    x = np.transpose(x, (0, 2, 3, 1))
    return x, label
