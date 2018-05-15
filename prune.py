from VGG16Cifar10 import VGG16Cifar10
from VGG16SEBlock import VGG16SEBlock
from ResNet20 import ResNet20
from ResNet20SEBlock import ResNet20SEBlock
import tensorflow as tf
import numpy as np
import os
from util import Cifar10Dataset
import sys
import filter_reduce as fr


flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("dataset", None, "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("testset_size", 32, "testset size [32]")
flags.DEFINE_float("l2_lambda", 0.01, "l2 term lambda")
flags.DEFINE_string("model_name", "VGG16Cifar10", "model to train")
FLAGS = flags.FLAGS

def prune(batch_size, epoch_num, data_set, learning_rate, testset_size, checkpoint_dir, model, ues_regularizer=False):
    train_step = model.get_train_step(learning_rate, ues_regularizer)
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        load_result = model.load_weight(sess, saver, checkpoint_dir)
        if load_result == True:
            print("Checkpoint load success!")
        else:
            print("No checkpoint file,weight inited!")

        loss, acc = sess.run([model.loss, model.accaury],
                             feed_dict={model.x: data_set.test_x, model.y_: data_set.test_label, model.isTrain: False})
        print("Model init stat: loss is {},accuary is {}".format(loss, acc))


        loss, acc = sess.run([model.loss, model.accaury],
                             feed_dict={model.x: data_set.test_x, model.y_: data_set.test_label, model.isTrain: False})
        print("Model after pruning: loss is {},accuary is {}".format(loss, acc))


def thinet_channel_select(sess,input_channel,filters):
    input_channel_num = input_channel.shape[1]
    input_channel_size = input_channel.shape[2]
    filter_num = filters.shape[3]
    input_channel = np.pad(input_channel,((0,0), (1,1),(1,1),(0,0)), 'constant')
    print(input_channel.shape)
    prune_sample = np.array()

def seblock_channel_select(channel_weight,compress_rate):
    """

    :param channel_weight:numpy数组
        seblock的输出
    :param compress_rate:float
        压缩率
    :return numpy数组
        返回需要剪枝的通道index，index从大到小（方便剪枝）:
    """
    channel_weight = np.reshape(channel_weight,(-1,channel_weight.shape[3]))
    channel_weight = np.sum(channel_weight,axis=0)
    prune_num = int(channel_weight.shape[0]*compress_rate)
    channel_weight = channel_weight.argsort()[0:prune_num]
    channel_weight.sort()
    channel_weight = channel_weight[::-1]
    return channel_weight

def seblock_prune(sess,pre_filter,next_filter,seblock_filter,bn_filter,seblock_output,compress_rate):
    prune_index = seblock_channel_select(seblock_output, compress_rate)
    # for index in prune_index:



def main(_):
    batch_size = FLAGS.batch_size
    epoch_num = FLAGS.epoch
    dataset_path =  FLAGS.dataset
    learning_rate = FLAGS.learning_rate
    testset_size = FLAGS.testset_size
    l2_lambda = FLAGS.l2_lambda
    checkpoint_dir = FLAGS.checkpoint_dir

    batch_size = FLAGS.batch_size
    epoch_num = FLAGS.epoch
    dataset_path = FLAGS.dataset
    learning_rate = FLAGS.learning_rate
    testset_size = FLAGS.testset_size
    l2_lambda = FLAGS.l2_lambda
    checkpoint_dir = FLAGS.checkpoint_dir
    model_name = FLAGS.model_name
    ues_regularizer = False

    cifar10 = Cifar10Dataset(dataset_path)
    cifar10.load_train_data()
    print("Train data load success,train set shape:{}".format(cifar10.train_x.shape))
    cifar10.load_test_data()
    print("Test data load success,test set shape:{}".format(cifar10.test_x.shape))
    model = None
    if model_name == "VGG16SEBlock":
        model = VGG16SEBlock(l2_lambda)
        ues_regularizer = True
    elif model_name == "VGG16Cifar10":
        model = VGG16Cifar10(l2_lambda)
        ues_regularizer = True
    elif model_name == "ResNet20":
        model = ResNet20()
    elif model_name == "ResNet20SEBlock":
        model = ResNet20SEBlock()
    model.build_model()
    print("Model build success!")

    prune(batch_size, epoch_num, cifar10, learning_rate, testset_size, checkpoint_dir, model, ues_regularizer)

if __name__ == '__main__':
  tf.app.run()