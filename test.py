
import os
from glob import glob
import util
from util import Cifar10Dataset
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from VGG16Cifar10 import VGG16Cifar10
vgg = VGG16Cifar10(0.5)
vgg.build_model()
input = np.zeros((8,32,32,3))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(vgg.output2,feed_dict={vgg.x:input,vgg.isTrain:False})
    print(type(result))




