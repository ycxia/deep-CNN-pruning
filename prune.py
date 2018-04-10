from VGG16Cifar10 import VGG16Cifar10
import tensorflow as tf
import numpy as np
import os
from util import Cifar10Dataset
import sys


flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("dataset", None, "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("testset_size", 32, "testset size [32]")
flags.DEFINE_float("l2_lambda", 0.01, "l2 term lambda")
FLAGS = flags.FLAGS

def train_vgg_cifar10(batch_size, epoch_num, dataset_path, learning_rate, testset_size, l2_lambda, checkpoint_dir):
    weight_saver_dir = os.path.join(checkpoint_dir,'vgg16_cifar_epoch')

    cifar10 = Cifar10Dataset(dataset_path)
    cifar10.load_train_data()
    print("Train data load success,train set shape:{}".format(cifar10.train_x.shape))
    cifar10.load_test_data()
    print("Test data load success,test set shape:{}".format(cifar10.test_x.shape))
    load_result = cifar10.load_prune_data()
    if load_result==False:
        print("Prune data load fail,exit!")
        return

    vgg = VGG16Cifar10(l2_lambda)
    vgg.build_model()
    train_step = vgg.get_train_step(learning_rate,ues_regularizer=True)
    print("Model build success!")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        load_result = vgg.load_weight(sess,saver,weight_saver_dir)
        if(load_result == False):
            print("Weight load fail!")
            return
        print("Weight load success，pruning start!")



def main(_):
    batch_size = FLAGS.batch_size
    epoch_num = FLAGS.epoch
    dataset_path = os.path.join(sys.path[0],'data', FLAGS.dataset)
    learning_rate = FLAGS.learning_rate
    testset_size = FLAGS.testset_size
    l2_lambda = FLAGS.l2_lambda
    checkpoint_dir = FLAGS.checkpoint_dir

    train_vgg_cifar10(batch_size, epoch_num, dataset_path, learning_rate, testset_size, l2_lambda, checkpoint_dir)

if __name__ == '__main__':
  tf.app.run()