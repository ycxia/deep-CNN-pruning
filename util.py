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
    # x = (x - np.mean(x)) / np.std(x)
    x = x/255.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        label = sess.run(label)
    show = np.array([])
    for i in range(x.shape[0]//3072):
        image = x[i * 3072:(i + 1) * 3072]
        image = np.reshape(image, (32, 32, 3), 'F')
        show = np.append(show, image)
    x = np.reshape(show, (-1, 32, 32, 3))
    return x, label
