import pickle
import numpy as np
import os
from glob import glob
from imgaug import augmenters as iaa


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class Cifar10Dataset:
    def __init__(self,dir):
        self.dir = dir
        self.train_label = None
        self.train_x = None
        self.test_x = None
        self.test_label = None
        self.prune_x = None
        self.seq = iaa.Sequential([
            iaa.Pad(px=4,keep_size=False),
            # iaa.Crop(px=(0, 4)),  # crop images from each side by 0 to 4px (randomly chosen)
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        ])
        self.prune_data_simple_num = 100 #剪枝时每个类取多少样本
    def load_train_data(self):
        file_list = glob(os.path.join(self.dir, 'data*'))
        self.train_x = np.array([])
        self.train_label = np.array([])
        for file in file_list:
            dict = unpickle(file)
            self.train_x = np.concatenate((self.train_x,dict[b'data'].flatten()))
            self.train_label = np.concatenate((self.train_label,np.array(dict[b'labels'])))
        self.train_x = self._data_reshape(self.train_x)


    def load_test_data(self):
        file_dir = os.path.join(self.dir, 'test_batch')
        dict = unpickle(file_dir)
        self.test_x = dict[b'data'].flatten()
        self.test_x = self._data_reshape(self.test_x)
        self.test_label = dict[b'labels']
        self.test_x = self.normalize(self.test_x)

    def _data_reshape(self,data):
        data = np.reshape(data, (-1, 3, 32, 32))
        data = np.transpose(data, (0, 2, 3, 1))
        return data

    def shuffle_train_data(self):
        permutation = np.random.permutation(self.train_x.shape[0])
        self.train_x = self.train_x[permutation, :, :]
        self.train_label = self.train_label[permutation]

    def load_prune_data(self):
        self.prune_x = []
        all_count = 0
        class_count = np.zeros(self.prune_data_simple_num, dtype=int)
        for (image,label) in zip(self.train_x,self.train_label):
            label = int(label)
            if class_count[label] < self.prune_data_simple_num:
                all_count += 1
                class_count[label] += 1
                self.prune_x.append(image)
                if all_count == 10*self.prune_data_simple_num:
                    self.prune_x = np.array(self.prune_x)
                    self.prune_x = np.reshape(self.prune_x, (-1, 32, 32, 3))
                    return True
        return False

    def data_argument(self,x):
        N = x.shape[0]
        x = self.seq.augment_images(x)
        x = np.array(x)
        random = np.random.randint(9, size=(N, 2))
        xx = []
        for i in range(N):
            xx.append(x[i, random[i][0]:random[i][0] + 32, random[i][1]:random[i][1] + 32])
        xx = np.array(xx)
        return xx

    def normalize(self,x):
        x = x/255.0
        x = (x-[0.4914, 0.4822, 0.4465])/[0.2023, 0.1994, 0.2010]
        return x

