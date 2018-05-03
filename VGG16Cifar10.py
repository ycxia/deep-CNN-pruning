import tensorflow as tf
import numpy as np

class VGG16Cifar10:
    def __init__(self,lbda):
        self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.y_ = tf.placeholder(tf.int64, shape=(None,))
        self.isTrain = tf.placeholder(tf.bool)
        self.regularizer = tf.contrib.layers.l2_regularizer(lbda)
        self.filters = []
        self.dense = []
        self.bais = []
        self.filters.append(self.get_variable("filter1",shape=[3,3,3,64]))
        self.filters.append(self.get_variable("filter2",shape=[3,3,64,64]))
        self.filters.append(self.get_variable("filter3",shape=[3, 3, 64, 128]))
        self.filters.append(self.get_variable("filter4",shape=[3, 3, 128, 128]))
        self.filters.append(self.get_variable("filter5",shape=[3, 3, 128, 256]))
        self.filters.append(self.get_variable("filter6",shape=[3, 3, 256, 256]))
        self.filters.append(self.get_variable("filter7",shape=[3, 3, 256, 256]))
        self.filters.append(self.get_variable("filter8",shape=[3, 3, 256, 512]))
        self.filters.append(self.get_variable("filter9",shape=[3, 3, 512, 512]))
        self.filters.append(self.get_variable("filter10",shape=[3, 3, 512, 512]))
        self.filters.append(self.get_variable("filter11",shape=[3, 3, 512, 512]))
        self.filters.append(self.get_variable("filter12",shape=[3, 3, 512, 512]))
        self.filters.append(self.get_variable("filter13",shape=[3, 3, 512, 512]))

        self.dense.append(self.get_variable("dense1",shape=[512,512]))
        self.bais.append(tf.zeros(name="b1",shape=[1,512]))
        self.dense.append(self.get_variable("dense2",shape=[512,10]))
        self.bais.append(tf.zeros(name="b2", shape=[1, 10]))



    def build_model(self):
        self.output1 = self.conv2d_with_relu(self.x, self.filters[0])
        self.output1 = tf.layers.dropout(self.output1,0.3,training=self.isTrain)
        self.output2 = self.conv2d_with_relu(self.output1, self.filters[1])
        polled = tf.nn.max_pool(self.output2, [1,2,2,1], [1,2,2,1],'VALID')

        self.output3 = self.conv2d_with_relu(polled, self.filters[2])
        self.output3 = tf.layers.dropout(self.output3, 0.4,training=self.isTrain)
        self.output4 = self.conv2d_with_relu(self.output3, self.filters[3])
        polled = tf.nn.max_pool(self.output4, [1,2,2,1], [1,2,2,1], 'VALID')

        self.output5 = self.conv2d_with_relu(polled, self.filters[4])
        self.output5 = tf.layers.dropout(self.output5, 0.4,training=self.isTrain)
        self.output6 = self.conv2d_with_relu(self.output5, self.filters[5])
        self.output6 = tf.layers.dropout(self.output6, 0.4,training=self.isTrain)
        self.output7 = self.conv2d_with_relu(self.output6, self.filters[6])
        polled = tf.nn.max_pool(self.output7, [1,2,2,1], [1, 2, 2, 1], 'VALID')

        self.output8 = self.conv2d_with_relu(polled, self.filters[7])
        self.output8 = tf.layers.dropout(self.output8, 0.4,training=self.isTrain)
        self.output9 = self.conv2d_with_relu(self.output8, self.filters[8])
        self.output9 = tf.layers.dropout(self.output9, 0.4,training=self.isTrain)
        self.output10 = self.conv2d_with_relu(self.output9, self.filters[9])
        polled = tf.nn.max_pool(self.output10, [1,2,2,1], [1, 2, 2, 1], 'VALID')

        self.output11 = self.conv2d_with_relu(polled, self.filters[10])
        self.output11 = tf.layers.dropout(self.output11, 0.4,training=self.isTrain)
        self.output12 = self.conv2d_with_relu(self.output11, self.filters[11])
        self.output12 = tf.layers.dropout(self.output12, 0.4,training=self.isTrain)
        self.output13 = self.conv2d_with_relu(self.output12, self.filters[12])
        polled = tf.nn.max_pool(self.output13, [1,2,2,1], [1, 2, 2, 1], 'VALID')

        polled = tf.reshape(polled,[-1,512])
        polled = tf.layers.dropout(polled, 0.4,training=self.isTrain)
        fc1 = self.fc(polled,self.dense[0],self.bais[0])
        fc1 = tf.layers.batch_normalization(fc1,training=self.isTrain)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1,0.5)
        self.y = self.fc(fc1,self.dense[1],self.bais[1])
        self.yy = tf.nn.softmax(self.y)
        correct_prediction = tf.equal(tf.argmax(self.yy,1), self.y_)
        self.accaury = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_,
            logits=self.y))

    def conv2d_with_relu(self, input, filter):
        output = tf.nn.conv2d(input, filter, [1, 1, 1, 1], 'SAME')
        output = tf.layers.batch_normalization(output,training=self.isTrain)
        output = tf.nn.relu(output)
        return output

    def fc(self,input,dense,bais):
        return tf.matmul(input, dense) + bais

    def get_train_step(self, learning_rate, ues_regularizer=False):
        self.loss = self.cross_entropy
        if(ues_regularizer):
            self.loss += tf.reduce_sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            )
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        return train_step

    def get_variable(self,name,shape):
        return tf.get_variable(name=name,
                               shape=shape,
                               initializer=tf.glorot_normal_initializer(),
                               regularizer=self.regularizer)
    def load_weight(self,sess,saver,weight_saver_dir):
        try:
            saver.restore(sess, weight_saver_dir)
            return True
        except BaseException:
            sess.run(tf.global_variables_initializer())
            return False

    def train(self,sess, batch_size, epoch_num,data_set, train_step, testset_size, weight_saver_dir, saver):
        loss, acc = sess.run([self.loss, self.accaury],
                             feed_dict={self.x: data_set.test_x, self.y_: data_set.test_label, self.isTrain: False})
        print("Model init stat: loss is {},accuary is {}".format(loss, acc))
        max_acc = acc
        train_data_size = len(data_set.train_label)
        batch_num = train_data_size // batch_size
        for epoch in range(epoch_num):
            # 每个epoch都打乱数据顺序
            random_order = np.random.permutation(train_data_size)
            for i in range(batch_num):
                batch_index = random_order[i*batch_size : min(i*batch_size+batch_size,train_data_size)]
                batch_x = data_set.train_x[batch_index]
                batch_label = data_set.train_label[batch_index]
                sess.run(train_step, feed_dict={self.x: batch_x, self.y_: batch_label, self.isTrain:True})
                if i % 100 == 0:
                    loss,acc = sess.run([self.loss,self.accaury], feed_dict={self.x: data_set.test_x[0:testset_size], self.y_: data_set.test_label[0:testset_size], self.isTrain:False})
                    train_loss, train_acc = sess.run([self.loss,self.accaury], feed_dict={self.x: batch_x, self.y_: batch_label, self.isTrain:False})
                    print("{}/{} batch: loss is {},acc is {}. on train set:{},{}".format(i,batch_num,loss,acc,train_loss,train_acc))
            loss, acc = sess.run([self.loss, self.accaury], feed_dict={self.x: data_set.test_x, self.y_: data_set.test_label, self.isTrain:False})
            print("{} epoch: loss is {},accuary is {}".format(epoch,loss,acc))
            if acc>max_acc:
                saver.save(sess, weight_saver_dir)
                max_acc = acc
                print("{} epoch weight save success!".format(epoch))
        print("Training end!")
