import tensorflow as tf
import numpy as np

class VGG16SEBlock:
    def __init__(self,lbda):
        channel_nums = [64,64,128,128,256,256,256,512,512,512,512,512]
        self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.y_ = tf.placeholder(tf.int64, shape=(None,))
        self.isTrain = tf.placeholder(tf.bool)
        self.regularizer = tf.contrib.layers.l2_regularizer(lbda)
        self.filters = []
        self.dense = []
        self.bais = []
        self.seblock_dense1=[]
        self.seblock_bais1 = []
        self.seblock_dense2 = []
        self.seblock_bais2 = []
        self.seblock_ouputs = []
        for (num,i) in zip(channel_nums,range(len(channel_nums))):
            mid_dense_num = num//8
            self.seblock_dense1.append(self.get_variable("seblock_dense1_"+str((i+1)),shape=[num,mid_dense_num]))
            self.seblock_dense2.append(self.get_variable("seblock_dense2_" + str((i+1)), shape=[mid_dense_num,num]))
            self.seblock_bais1.append(tf.zeros(name="seblock_bais1_" + str((i+1)), shape=[1, mid_dense_num]))
            self.seblock_bais2.append(tf.zeros(name="seblock_bais2_" + str((i+1)), shape=[1, num]))
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
        self.output1 = self.conv2d_with_relu_seblock(self.x, 0)
        self.output1 = tf.layers.dropout(self.output1,0.3,training=self.isTrain)
        self.output2 = self.conv2d_with_relu_seblock(self.output1, 1)
        pooled = tf.nn.max_pool(self.output2, [1,2,2,1], [1,2,2,1],'VALID')

        self.output3 = self.conv2d_with_relu_seblock(pooled, 2)
        self.output3 = tf.layers.dropout(self.output3, 0.4,training=self.isTrain)
        self.output4 = self.conv2d_with_relu_seblock(self.output3, 3)
        pooled = tf.nn.max_pool(self.output4, [1,2,2,1], [1,2,2,1], 'VALID')

        self.output5 = self.conv2d_with_relu_seblock(pooled, 4)
        self.output5 = tf.layers.dropout(self.output5, 0.4,training=self.isTrain)
        self.output6 = self.conv2d_with_relu_seblock(self.output5, 5)
        self.output6 = tf.layers.dropout(self.output6, 0.4,training=self.isTrain)
        self.output7 = self.conv2d_with_relu_seblock(self.output6, 6)
        pooled = tf.nn.max_pool(self.output7, [1,2,2,1], [1, 2, 2, 1], 'VALID')

        self.output8 = self.conv2d_with_relu_seblock(pooled, 7)
        self.output8 = tf.layers.dropout(self.output8, 0.4,training=self.isTrain)
        self.output9 = self.conv2d_with_relu_seblock(self.output8, 8)
        self.output9 = tf.layers.dropout(self.output9, 0.4,training=self.isTrain)
        self.output10 = self.conv2d_with_relu_seblock(self.output9, 9)
        pooled = tf.nn.max_pool(self.output10, [1,2,2,1], [1, 2, 2, 1], 'VALID')

        self.output11 = self.conv2d_with_relu_seblock(pooled, 10)
        self.output11 = tf.layers.dropout(self.output11, 0.4,training=self.isTrain)
        self.output12 = self.conv2d_with_relu_seblock(self.output11, 11)
        self.output12 = tf.layers.dropout(self.output12, 0.4,training=self.isTrain)
        self.output13 = self.conv2d_with_relu(self.output12, 12)
        pooled = tf.nn.max_pool(self.output13, [1,2,2,1], [1, 2, 2, 1], 'VALID')

        pooled = tf.layers.flatten(pooled)
        pooled = tf.layers.dropout(pooled, 0.4,training=self.isTrain)
        fc1 = self.fc(pooled,self.dense[0],self.bais[0])
        fc1 = tf.layers.batch_normalization(fc1)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1,0.5)
        self.y = self.fc(fc1,self.dense[1],self.bais[1])
        self.yy = tf.nn.softmax(self.y)
        correct_prediction = tf.equal(tf.argmax(self.yy,1), self.y_)
        self.accaury = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_,
            logits=self.y))

    def conv2d_with_relu(self, input, i):
        output = tf.nn.conv2d(input, self.filters[i], [1, 1, 1, 1], 'SAME')
        output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)
        return output

    def conv2d_with_relu_seblock(self, input, i):

        output = tf.nn.conv2d(input, self.filters[i], [1, 1, 1, 1], 'SAME')
        # output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)
        self.se_block(output,i)
        return output

    def se_block(self,input,i):
        channel_size = input.shape[1]
        output = tf.layers.average_pooling2d(input, [channel_size, channel_size], 1, 'valid')
        output = tf.layers.flatten(output)
        output = self.fc(output,self.seblock_dense1[i],self.seblock_bais1[i])
        output = tf.nn.relu(output)
        output = self.fc(output, self.seblock_dense2[i], self.seblock_bais2[i])
        output = tf.nn.sigmoid(output)
        self.seblock_ouputs.append(output)
        output = tf.expand_dims(output, 1)
        output = tf.expand_dims(output, 1)
        output = input*output
        return output


    def fc(self,input,dense,bais):
        return tf.matmul(input, dense) + bais

    def get_train_step(self, learning_rate, ues_regularizer=False):
        self.loss = self.cross_entropy
        if(ues_regularizer):
            self.loss += tf.reduce_sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            )
        return tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

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

