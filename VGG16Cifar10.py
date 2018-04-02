import tensorflow as tf

class VGG16Cifar10:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.y_ = tf.placeholder(tf.int64, shape=(None, 10))
        self.filters = []
        self.dense = []
        self.bais = []

        # self.filters.append(tf.Variable(np.random.rand(3, 3, 3, 64), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 64, 64), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 64, 128), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 128, 128), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 128, 256), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 256, 256), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 256, 256), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 256, 512), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 512, 512), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 512, 512), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 512, 512), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 512, 512), dtype=np.float32))
        # self.filters.append(tf.Variable(np.random.rand(3, 3, 512, 512), dtype=np.float32))
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
        self.output2 = self.conv2d_with_relu(self.output1, self.filters[1])
        polled = tf.nn.max_pool(self.output2, [1,2,2,1], [1,2,2,1],'VALID')
        self.output3 = self.conv2d_with_relu(polled, self.filters[2])
        self.output4 = self.conv2d_with_relu(self.output3, self.filters[3])
        polled = tf.nn.max_pool(self.output4, [1,2,2,1], [1,2,2,1], 'VALID')
        self.output5 = self.conv2d_with_relu(polled, self.filters[4])
        self.output6 = self.conv2d_with_relu(self.output5, self.filters[5])
        self.output7 = self.conv2d_with_relu(self.output6, self.filters[6])
        polled = tf.nn.max_pool(self.output7, [1,2,2,1], [1, 2, 2, 1], 'VALID')
        self.output8 = self.conv2d_with_relu(polled, self.filters[7])
        self.output9 = self.conv2d_with_relu(self.output8, self.filters[8])
        self.output10 = self.conv2d_with_relu(self.output9, self.filters[9])
        polled = tf.nn.max_pool(self.output10, [1,2,2,1], [1, 2, 2, 1], 'VALID')
        self.output11 = self.conv2d_with_relu(polled, self.filters[10])
        self.output12 = self.conv2d_with_relu(self.output11, self.filters[11])
        self.output13 = self.conv2d_with_relu(self.output12, self.filters[12])
        polled = tf.nn.max_pool(self.output13, [1,2,2,1], [1, 2, 2, 1], 'VALID')

        polled = tf.reshape(polled,[-1,512])
        fc1 = self.fc(polled,self.dense[0],self.bais[0])
        fc1 = tf.nn.relu(fc1)
        # fc1 = tf.layers.dense(polled,512,tf.nn.relu)
        fc1 = tf.nn.dropout(fc1,0.5)

        self.y = self.fc(fc1,self.dense[1],self.bais[1])
        # self.y = tf.layers.dense(fc1,10)
        # self.fc1 = self.fc_relu_drop(polled)
        # self.y = tf.layers.dense(inputs=self.fc1, units=10, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
        self.yy = tf.nn.softmax(self.y)
        correct_prediction = tf.equal(tf.argmax(self.yy,1), tf.argmax(self.y_,1))
        self.accaury = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_,
            logits=self.y))

        # self.add_weight_to_collection()

    def conv2d_with_relu(self, input, filter):
        output = tf.nn.conv2d(input, filter, [1, 1, 1, 1], 'SAME')
        output = tf.nn.relu(output)
        return output

    def fc(self,input,dense,bais):
        return tf.matmul(input, dense) + bais

    def get_train_step(self, learning_rate,lbd):
        # regularizer = tf.contrib.layers.l2_regularizer(scale=lbd)
        # self.reg_term = tf.contrib.layers.apply_regularization(regularizer)
        return tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

    def get_variable(self,name,shape):
        return tf.get_variable(name=name,shape=shape,initializer=tf.glorot_normal_initializer())

    def add_weight_to_collection(self):
        for filter in self.filters:
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, filter)
        for dense in self.dense:
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, dense)
