
import os
from glob import glob
import util
from util import Cifar10Dataset
import pickle
import matplotlib.pyplot as plt
import numpy as np
from ResNet20 import ResNet20
import tensorflow as tf
import filter_reduce as fr

cifar10 = Cifar10Dataset("/home/wxj/下载/dataset/cifar-10-batches-py")
cifar10.load_test_data()
print("Test data load success,test set shape:{}".format(cifar10.test_x.shape))
model = ResNet20()
model.build_model()
with tf.variable_scope("",reuse=True):
    f1 = tf.get_variable("block1/conv2d/kernel")
    f2 = tf.get_variable("block1/conv2d_1/kernel")
    bn_gm = tf.get_variable("block1/batch_normalization/gamma")
    bn_mean = tf.get_variable("block1/batch_normalization/moving_mean")
    bn_var = tf.get_variable("block1/batch_normalization/moving_variance")
    bn_beta = tf.get_variable("block1/batch_normalization/beta")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
acc = sess.run(model.accaury,feed_dict={model.x:cifar10.test_x[:100],model.y_:cifar10.test_label[:100], model.isTrain:False})
print(acc)
shape1 = list((int(f1.shape[i]) for i in range(4)))
shape2 = list((int(f2.shape[i]) for i in range(4)))
fr.reduceFilter(f1, sess, 0, 3, shape1)
fr.reduceFilter(f2, sess, 0, 2, shape2)
fr.reduceFilter(f1, sess, 0, 3, shape1)
fr.reduceFilter(f2, sess, 0, 2, shape2)
fr.reduceFilter(f1, sess, 0, 3, shape1)
fr.reduceFilter(f2, sess, 0, 2, shape2)
fr.reduceFilter(f1, sess, 0, 3, shape1)
fr.reduceFilter(f2, sess, 0, 2, shape2)

fr.reduceFilter(bn_gm, sess, 0, 0, [16])
fr.reduceFilter(bn_mean, sess, 0, 0, [16])
fr.reduceFilter(bn_var, sess, 0, 0, [16])
fr.reduceFilter(bn_beta, sess, 0, 0, [16])
fr.reduceFilter(bn_gm, sess, 0, 0, [15])
fr.reduceFilter(bn_mean, sess, 0, 0, [15])
fr.reduceFilter(bn_var, sess, 0, 0, [15])
fr.reduceFilter(bn_beta, sess, 0, 0, [15])
fr.reduceFilter(bn_gm, sess, 0, 0, [14])
fr.reduceFilter(bn_mean, sess, 0, 0, [14])
fr.reduceFilter(bn_var, sess, 0, 0, [14])
fr.reduceFilter(bn_beta, sess, 0, 0, [14])
fr.reduceFilter(bn_gm, sess, 0, 0, [13])
fr.reduceFilter(bn_mean, sess, 0, 0, [13])
fr.reduceFilter(bn_var, sess, 0, 0, [13])
fr.reduceFilter(bn_beta, sess, 0, 0, [13])



acc = sess.run(model.accaury, feed_dict={model.x: cifar10.test_x[:100], model.y_: cifar10.test_label[:100], model.isTrain:False})
print(acc)
sess.close()



# tensorboard_dir = 'tensorboard/ResNet20'  # 保存目录
# if not os.path.exists(tensorboard_dir):
#     os.makedirs(tensorboard_dir)
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter(tensorboard_dir)
#     writer.add_graph(sess.graph)





