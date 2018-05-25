
import os
from util import Cifar10Dataset
import pickle
import matplotlib.pyplot as plt
import numpy as np
from ResNet20 import ResNet20
from ResNet20SEBlock import ResNet20SEBlock
from VGG16SEBlock import VGG16SEBlock
from VGG16 import VGG16
import tensorflow as tf
import filter_reduce as fr

# cifar10 = Cifar10Dataset("/home/wxj/下载/dataset/cifar-10-batches-py")
# cifar10.load_test_data()
# print("Test data load success,test set shape:{}".format(cifar10.test_x.shape))
# model = ResNet20()
# model.build_model()
#
# acc = sess.run(model.accaury, feed_dict={model.x: cifar10.test_x[:100], model.y_: cifar10.test_label[:100], model.isTrain:False})
# print(acc)
# sess.close()


model = VGG16(0.0001)
model.build_model()
saver = tf.train.Saver(max_to_keep=1)
data = Cifar10Dataset("data/cifar-10-batches-py")
data.load_test_data()
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # saver.save(sess,save_path="tensorboard/checkpoint/vggtest")
    model.load_weight(sess,saver,"tensorboard/checkpoint/vgg16_0.88")
    output = sess.run(model.seblock_output[0],feed_dict={model.x:data.test_x,model.y:data.test_label,model.isTrain:False})
    print(output[0])

# tensorboard_dir = 'tensorboard/VGG16SEBlock'  # 保存目录
# if not os.path.exists(tensorboard_dir):
#     os.makedirs(tensorboard_dir)
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter(tensorboard_dir)
#     writer.add_graph(sess.graph)





