import tensorflow as tf
import numpy as np

class VGG:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.y_ = tf.placeholder(tf.float32, shape=(None, 10))
        self.filters = []
        self.filters.append(tf.Variable(np.random.rand(3, 3, 3, 64), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 64, 64), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 64, 128), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 128, 128), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 128, 256), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 256, 256), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 256, 256), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 256, 512), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 512, 512), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 512, 512), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 512, 512), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 512, 512), dtype=np.float32))
        self.filters.append(tf.Variable(np.random.rand(3, 3, 512, 512), dtype=np.float32))

    def build_model(self):
        self.output1 = self.conv2d_with_relu(self.x, self.filters[0])
        self.output2 = self.conv2d_with_relu(self.output1, self.filters[1])
        polled = tf.nn.max_pool(self.output2, [1,2,2,1], 'VALID')
        self.output3 = self.conv2d_with_relu(polled, self.filters[2])
        self.output4 = self.conv2d_with_relu(self.output3, self.filters[3])
        polled = tf.nn.max_pool(self.output4, [1,2,2,1], 'VALID')
        self.output5 = self.conv2d_with_relu(polled, self.filters[4])
        self.output6 = self.conv2d_with_relu(self.output5, self.filters[5])
        self.output7 = self.conv2d_with_relu(self.output6, self.filters[6])
        polled = tf.nn.max_pool(self.output7, [1, 2, 2, 1], 'VALID')
        self.output8 = self.conv2d_with_relu(polled, self.filters[7])
        self.output9 = self.conv2d_with_relu(self.output8, self.filters[8])
        self.output10 = self.conv2d_with_relu(self.output9, self.filters[9])
        polled = tf.nn.max_pool(self.output10, [1, 2, 2, 1], 'VALID')
        self.output11 = self.conv2d_with_relu(polled, self.filters[10])
        self.output12 = self.conv2d_with_relu(self.output11, self.filters[11])
        self.output13 = self.conv2d_with_relu(self.output12, self.filters[12])
        polled = tf.nn.max_pool(self.output13, [1, 2, 2, 1], 'VALID')

        polled = tf.reshape(polled,[None,20])
        fc1 = self.fc_relu_drop(polled)
        y = tf.layers.dense(inputs=fc1, units=10)
        y = tf.nn.softmax(y)
        correct_prediction = tf.equal(tf.arg_max(y,1), self.y_)
        self.accaury = tf.reduce_mean(tf.cast(correct_prediction), tf.float32)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_,
            logits=y)

    def conv2d_with_relu(self, input, filter):
        output = tf.nn.conv(input, filter, [1, 1, 1, 1], 'SAME')
        output = tf.nn.relu(output)
        return output

    def fc_relu_drop(self, input, filter):
        output = tf.layers.dense(inputs=input, units=512, activation=tf.nn.relu)
        output = tf.nn.dropout(output,0.5)
        return output

    def get_train_step(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)