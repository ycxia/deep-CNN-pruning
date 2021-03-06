import tensorflow as tf
class ResNet20SEBlock:
    def __init__(self, lbda):
        self.x = tf.placeholder(tf.float32,[None,32,32,3])
        self.y_ = tf.placeholder(tf.int64,[None,])
        self.isTrain = tf.placeholder(tf.bool)
        self.regularizer = tf.contrib.layers.l2_regularizer(lbda)
        self.seblock_ouput = []

    def build_model(self):
        output = tf.layers.conv2d(self.x, 16, 3, 1, 'same',use_bias=False, kernel_regularizer=self.regularizer)
        output = tf.layers.batch_normalization(output,training=self.isTrain)
        output = tf.nn.relu(output)
        output = self.residual_block(output, "block1", 3, 16, False)
        output = self.residual_block(output, "block2", 3, 32)
        output = self.residual_block(output, "block3", 3, 64)

        output = tf.layers.average_pooling2d(output, 8, 1, 'valid')
        output = tf.layers.flatten(output)
        self.y = tf.layers.dense(output, 10, kernel_regularizer=self.regularizer)
        self.yy = tf.nn.softmax(self.y)
        correct_prediction = tf.equal(tf.argmax(self.yy, 1), self.y_)
        self.accaury = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_,
            logits=self.y))

    def residual_block(self, x, name, res_num, output_num, differ_dim=True):
        output = x
        with tf.variable_scope(name):
            for i in range(res_num):
                name = "res_modules_" + str(i+1)
                if(i==0):
                    output = self.residual_model(output, output_num, name, differ_dim)
                else:
                    output = self.residual_model(output, output_num, name)
        return output

    def residual_model(self, x, output_num, name, differ_dim=False):
        with tf.variable_scope(name):
            if(differ_dim==True):
                output = tf.layers.conv2d(x, output_num, 3, 2, 'same',use_bias=False, kernel_regularizer=self.regularizer, name="conv_1")

                x = tf.layers.average_pooling2d(x, 2, 2, 'valid')
                padding = tf.constant([[0, 0], [0, 0], [0, 0], [output_num // 4, output_num // 4]])
                x = tf.pad(x, padding, "CONSTANT")
            else:
                output = tf.layers.conv2d(x, output_num, 3, 1, 'same',use_bias=False, kernel_regularizer=self.regularizer, name="conv_1")
            output = tf.layers.batch_normalization(output, training=self.isTrain, name="bn_1")
            output = self.se_block(output, "seblock_1")
            output = tf.nn.relu(output)
            output = tf.layers.conv2d(output, output_num, 3, 1, 'same',use_bias=False, kernel_regularizer=self.regularizer, name="conv_2")
            output = tf.layers.batch_normalization(output,training=self.isTrain, name="bn_2")
            output = tf.nn.relu(output+x)
            return output

    def get_train_step(self, lr, ues_regularizer):
        self.loss = self.cross_entropy
        if ues_regularizer==True:
            self.loss += tf.reduce_sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            )
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = tf.train.MomentumOptimizer(lr,0.9).minimize(self.loss)
        return train_op

    def load_weight(self,sess,saver,weight_saver_dir):
        try:
            saver.restore(sess, weight_saver_dir)
            return True
        except BaseException:
            sess.run(tf.global_variables_initializer())
            return False

    def se_block(self,input,name):
        with tf.variable_scope(name):
            channel_size = input.shape[1]
            channel_num = input.shape[3]
            output = tf.layers.average_pooling2d(input, [channel_size,channel_size], 1, 'valid')
            output = tf.layers.flatten(output)
            output = tf.layers.dense(output,channel_num//8,tf.nn.relu, kernel_regularizer=self.regularizer, name="dense_0")
            output = tf.layers.dense(output,channel_num,tf.nn.sigmoid, kernel_regularizer=self.regularizer, name="dense_1")
            self.seblock_ouput.append(output)
            output = tf.expand_dims(output, 1)
            output = tf.expand_dims(output, 1)
            output = input*output
            return output