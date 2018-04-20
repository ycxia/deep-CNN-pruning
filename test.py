
import os
from glob import glob
import util
from util import Cifar10Dataset
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from VGG16SEBlock import VGG16SEBlock

model = VGG16SEBlock(0.0001)
model.build_model()
tensorboard_dir = 'tensorboard/'   # 保存目录
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

writer = tf.summary.FileWriter(tensorboard_dir)
with tf.Session() as sess:

    writer.add_graph(sess.graph)



