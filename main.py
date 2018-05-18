from VGG16Cifar10 import VGG16Cifar10
from VGG16SEBlock import VGG16SEBlock
from ResNet20 import ResNet20
from ResNet20SEBlock import ResNet20SEBlock
import tensorflow as tf
from util import Cifar10Dataset
import numpy as np
import os


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

def train(batch_size, epoch_num, data_set, learning_rate, testset_size, checkpoint_dir, model, ues_regularizer=False):
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
        max_acc = max(acc, 0.85)
        train_data_size = len(data_set.train_label)
        batch_num = train_data_size // batch_size
        for epoch in range(epoch_num):
            # 每个epoch都打乱数据顺序
            random_order = np.random.permutation(train_data_size)
            for i in range(batch_num):
                batch_index = random_order[i * batch_size: min(i * batch_size + batch_size, train_data_size)]
                batch_x = data_set.train_x[batch_index]
                batch_label = data_set.train_label[batch_index]
                # 随机crop
                flip_image = batch_x[:len(batch_label)//2]
                batch_x[:len(batch_label)//2] = flip_image[:, :, ::-1, :]
                batch_x = np.pad(batch_x, ((0, 0), (4, 4), (4, 4), (0, 0)), 'constant')

                sess.run(train_step, feed_dict={model.x: batch_x, model.y_: batch_label, model.isTrain: True})
                if i % 100 == 0:
                    loss, acc = sess.run([model.loss, model.accaury], feed_dict={model.x: data_set.test_x[0:testset_size],
                                                                                 model.y_: data_set.test_label[0:testset_size],
                                                                                 model.isTrain: False})
                    train_loss, train_acc = sess.run([model.loss, model.accaury],
                                                     feed_dict={model.x: batch_x,
                                                                model.y_: batch_label,
                                                                model.isTrain: False})
                    print("{}/{} batch: loss is {},acc is {}. on train set:{},{}".format(i, batch_num, loss, acc,train_loss, train_acc))
            loss, acc = sess.run([model.loss, model.accaury],
                                 feed_dict={model.x: data_set.test_x, model.y_: data_set.test_label, model.isTrain: False})
            print("{} epoch: loss is {},accuary is {}".format(epoch, loss, acc))
            if acc > max_acc:
                saver.save(sess, "{}_{}".format(checkpoint_dir, '%.3f' % acc))
                max_acc = acc
                print("{} epoch weight save success!".format(epoch))
        print("Training end!")

def main(_):
    batch_size = FLAGS.batch_size
    epoch_num = FLAGS.epoch
    dataset_path =  FLAGS.dataset
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
    if model_name=="VGG16SEBlock":
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
    train(batch_size, epoch_num, cifar10, learning_rate, testset_size, checkpoint_dir, model, ues_regularizer)

if __name__ == '__main__':
  tf.app.run()