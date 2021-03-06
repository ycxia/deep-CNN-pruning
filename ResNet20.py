import tensorflow as tf
class ResNet20:
    def __init__(self, l2_lambda, l1_lambda):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,32,32,3])
        self.y_ = tf.placeholder(tf.int64,[None,])
        self.isTrain = tf.placeholder(tf.bool)
        self.l2_reg = tf.contrib.layers.l2_regularizer(l2_lambda)
        self.l1_reg = tf.contrib.layers.l1_regularizer(l1_lambda)

    def build_model(self):
        output = tf.layers.conv2d(self.x, 16, 3, 1, 'same', use_bias=False, kernel_regularizer=self.l2_reg)
        output = tf.layers.batch_normalization(output,training=self.isTrain, gamma_regularizer=self.l1_reg)
        output = tf.nn.relu(output)
        output = self.residual_block(output, "block1", 3, 16, False)
        output = self.residual_block(output, "block2", 3, 32)
        output = self.residual_block(output, "block3", 3, 64)

        output = tf.layers.average_pooling2d(output, 8, 1, 'valid')
        output = tf.layers.flatten(output)
        self.y = tf.layers.dense(output, 10, kernel_regularizer=self.l2_reg)
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
                if(i==0):
                    output = self.residual_model(output, output_num, differ_dim)
                else:
                    output = self.residual_model(output, output_num)
        return output

    def residual_model(self, x, output_num, differ_dim=False):
        if(differ_dim==True):
            output = tf.layers.conv2d(x, output_num, 3, 2, 'same', use_bias=False, kernel_regularizer=self.l2_reg)

            x = tf.layers.average_pooling2d(x, 2, 2, 'valid')
            padding = tf.constant([[0, 0], [0, 0], [0, 0], [output_num // 4, output_num // 4]])
            x = tf.pad(x, padding, "CONSTANT")
        else:
            output = tf.layers.conv2d(x, output_num, 3, 1, 'same', use_bias=False, kernel_regularizer=self.l2_reg)
        output = tf.layers.batch_normalization(output,training=self.isTrain, gamma_regularizer=self.l1_reg)
        output = tf.nn.relu(output)
        output = tf.layers.conv2d(output, output_num, 3, 1, 'same', use_bias=False, kernel_regularizer=self.l2_reg)
        output = tf.layers.batch_normalization(output,training=self.isTrain, gamma_regularizer=self.l1_reg)
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