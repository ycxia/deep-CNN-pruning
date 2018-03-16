import pickle
import numpy as np
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def data_read(train_file):
    x = np.array([])
    label = list()
    for file in train_file:
        dict = unpickle(file)
        # if x == None:
        #     x = dict[b'data']
        #     label = dict[b'labels']
        # else:
        x = np.concatenate((x,dict[b'data'].flatten()))
        for num in dict[b'labels']:
            label.append([num])
    return x, label
