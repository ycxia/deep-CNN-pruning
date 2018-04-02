import tensorflow as tf

class VGG16:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.y_ = tf.placeholder(tf.int64, shape=(None, 10))

    def build_model(self):
        output1 = self.conv2d_with_relu(self.x,64)
        output2 = self.conv2d_with_relu(output1, 64)
        output2 = self.max_pooling(output2)

        output3 = self.conv2d_with_relu(output2,128)
        output4 = self.conv2d_with_relu(output3, 128)
        output4 = self.max_pooling(output4)

        output5 = self.conv2d_with_relu(output4,256)
        output6 = self.conv2d_with_relu(output5, 256)
        output7 = self.conv2d_with_relu(output6, 256)
        output7 = self.max_pooling(output7)

        output8 = self.conv2d_with_relu(output7, 512)
        output9 = self.conv2d_with_relu(output8, 512)
        output10 = self.conv2d_with_relu(output9, 512)
        output10 = self.max_pooling(output10)

        output11 = self.conv2d_with_relu(output10, 512)
        output12 = self.conv2d_with_relu(output11, 512)
        output13 = self.conv2d_with_relu(output12, 512)
        output13 = self.max_pooling(output13)

        output13 = tf.reshape(output13, [-1, 512])
        fc1 = tf.layers.dense(output13,512,tf.nn.relu)
        fc1 = tf.layers.dropout(fc1,0.5)

        self.y = tf.layers.dense(fc1,10)
        self.yy = tf.nn.softmax(self.y)

        correct_prediction = tf.equal(tf.argmax(self.yy,1), tf.argmax(self.y_,1))
        self.accaury = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_,
            logits=self.y))


    def conv2d_with_relu(self, input,output_num):
        return tf.layers.conv2d(input,output_num,[3,3],[1,1],'same',activation=tf.nn.relu)

    def max_pooling(self,input):
        return tf.layers.max_pooling2d(input,[2,2],[2,2],'valid')


    def get_train_step(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)
