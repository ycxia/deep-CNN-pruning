import pickle
import numpy as np
import tensorflow as tf
import os
from glob import glob


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


class Cifar10Dataset:
    def __init__(self,dir):
        self.dir = dir
        self.train_label = None
        self.train_x = None
        self.test_x = None
        self.test_label = None
        self.prune_x = None
    def load_train_data(self):
        file_list = glob(os.path.join(self.dir, 'data*'))
        self.train_x = np.array([])
        self.train_label = np.array([])
        for file in file_list:
            dict = unpickle(file)
            self.train_x = np.concatenate((self.train_x,dict[b'data'].flatten()))
            self.train_label = np.concatenate((self.train_label,np.array(dict[b'labels'])))
        self.train_x = self._data_reshape(self.train_x)
        self.avg = np.mean(self.train_x,(0,1,2),dtype=np.float32)
        self.std = np.std(self.train_x,(0,1,2),dtype=np.float32)
        print("avg:{},std:{}".format(self.avg,self.std))
        self.train_x = (self.train_x-self.avg)/self.std

        # 数据增强
        # filp_data = self.train_x[:, :, ::-1, :]
        # self.train_x = np.concatenate((self.train_x, filp_data))
        # self.train_label = np.concatenate((self.train_label, self.train_label))
        # self.train_x = np.pad(self.train_x, ((0, 0), (4, 4), (4, 4), (0, 0)), 'constant')

    def load_test_data(self):
        file_dir = os.path.join(self.dir, 'test_batch')
        dict = unpickle(file_dir)
        self.test_x = dict[b'data'].flatten()
        self.test_x = self._data_reshape(self.test_x)
        self.test_label = dict[b'labels']
        self.test_x = (self.test_x - self.avg) / self.std

    def _data_reshape(self,data):
        data = np.reshape(data, (-1, 3, 32, 32))
        data = np.transpose(data, (0, 2, 3, 1))
        return data

    def shuffle_train_data(self):
        permutation = np.random.permutation(self.train_x.shape[0])
        self.train_x = self.train_x[permutation, :, :]
        self.train_label = self.train_label[permutation]

    def load_prune_data(self):
        self.prune_x = np.array([])
        all_count = 0;
        class_count = np.zeros(shape=(10))
        for image,label in self.train_x,self.train_label:
            if(class_count[label]<10):
                all_count+=1
                class_count[label]+=1
                self.prune_x = np.concatenate(self.prune_x,image)
                if(all_count==10*10):
                    self.prune_x = np.reshape(self.prune_x, (-1, 32, 32, 3))
                    return True
        return False



