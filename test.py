
import os
from glob import glob
import util
from util import Cifar10Dataset
import pickle
import matplotlib.pyplot as plt

cifar10 = Cifar10Dataset("data/cifar-10-batches-py")
cifar10.load_train_data()
print("data load success")
plt.imshow(cifar10.train_x[50444]) # 显示图片
plt.show()
print(len(cifar10.train_label))
print(cifar10.train_label[50444])





