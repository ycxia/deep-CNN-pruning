import tensorflow as tf

class VGG16:
    def __init__(self,lbda,prune_rate=0.3):
        # channel_nums = [64,64,128,128,256,256,256,512,512,512,512,512]
        self.prune_rate = prune_rate
        self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.y_ = tf.placeholder(tf.int64, shape=(None,))
        self.isTrain = tf.placeholder(tf.bool)
        self.regularizer = tf.contrib.layers.l2_regularizer(lbda)
        self.seblock_output = []

    def build_model(self):
        with tf.variable_scope("block_1"):
            self.output1 = self.conv2d_with_relu(self.x, 64-int(64*self.prune_rate), "conv_layer_1")
            self.output1 = tf.layers.dropout(self.output1,0.3,training=self.isTrain)
            self.output2 = self.conv2d_with_relu(self.output1, 64, "conv_layer_2")
            pooled = tf.nn.max_pool(self.output2, [1,2,2,1], [1,2,2,1],'VALID')

        with tf.variable_scope("block_2"):
            self.output3 = self.conv2d_with_relu(pooled, 128, "conv_layer_1")
            self.output3 = tf.layers.dropout(self.output3, 0.4,training=self.isTrain)
            self.output4 = self.conv2d_with_relu(self.output3, 128, "conv_layer_2")
            pooled = tf.nn.max_pool(self.output4, [1,2,2,1], [1,2,2,1], 'VALID')

        with tf.variable_scope("block_3"):
            self.output5 = self.conv2d_with_relu(pooled, 256, "conv_layer_1")
            self.output5 = tf.layers.dropout(self.output5, 0.4,training=self.isTrain)
            self.output6 = self.conv2d_with_relu(self.output5, 256, "conv_layer_2")
            self.output6 = tf.layers.dropout(self.output6, 0.4,training=self.isTrain)
            self.output7 = self.conv2d_with_relu(self.output6, 256, "conv_layer_3")
            pooled = tf.nn.max_pool(self.output7, [1,2,2,1], [1, 2, 2, 1], 'VALID')

        with tf.variable_scope("block_4"):
            self.output8 = self.conv2d_with_relu(pooled, 512, "conv_layer_1")
            self.output8 = tf.layers.dropout(self.output8, 0.4,training=self.isTrain)
            self.output9 = self.conv2d_with_relu(self.output8, 512, "conv_layer_2")
            self.output9 = tf.layers.dropout(self.output9, 0.4,training=self.isTrain)
            self.output10 = self.conv2d_with_relu(self.output9, 512, "conv_layer_3")
            pooled = tf.nn.max_pool(self.output10, [1,2,2,1], [1, 2, 2, 1], 'VALID')

        with tf.variable_scope("block_5"):
            self.output11 = self.conv2d_with_relu(pooled, 512, "conv_layer_1")
            self.output11 = tf.layers.dropout(self.output11, 0.4,training=self.isTrain)
            self.output12 = self.conv2d_with_relu(self.output11, 512, "conv_layer_2")
            self.output12 = tf.layers.dropout(self.output12, 0.4,training=self.isTrain)
            # 最后一层channelsize为1，不适合加seblock
            self.output13 = self.conv2d_with_relu(self.output12, 512,"conv_layer_3")
            pooled = tf.nn.max_pool(self.output13, [1,2,2,1], [1, 2, 2, 1], 'VALID')

        pooled = tf.layers.flatten(pooled)
        pooled = tf.layers.dropout(pooled, 0.4,training=self.isTrain)
        fc1 = tf.layers.dense(pooled, 512, kernel_regularizer=self.regularizer)
        fc1 = tf.layers.batch_normalization(fc1,training=self.isTrain)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1,0.5)
        self.y = tf.layers.dense(fc1, 10, kernel_regularizer=self.regularizer)
        self.yy = tf.nn.softmax(self.y)
        correct_prediction = tf.equal(tf.argmax(self.yy,1), self.y_)
        self.accaury = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_,
            logits=self.y))

    def conv3x3(self,input,output_num,name):
        return tf.layers.conv2d(input, filters=output_num, kernel_size=3, strides=1, padding='same', use_bias=False,
                         kernel_regularizer=self.regularizer,name=name)

    def conv2d_with_relu(self,input,output_num,name):
        with tf.variable_scope(name):
            output = self.conv3x3(input,output_num,"conv")
            output = tf.layers.batch_normalization(output,training=self.isTrain,name="bn")
            output = tf.nn.relu(output)
            return output

    def conv2d_with_relu_seblock(self, input, output_num,name):
        with tf.variable_scope(name):
            output = self.conv3x3(input,output_num,"conv")
            output = tf.layers.batch_normalization(output,training=self.isTrain,name="bn")
            # output = self.se_block(output,output_num,output_num//8,"seblock")
            output = tf.nn.relu(output)
            return output

    # def se_block(self,input,output_num,squeeze_size,name):
    #     with tf.variable_scope(name):
    #         channel_size = input.shape[1]
    #         output = tf.layers.average_pooling2d(input, [channel_size, channel_size], 1, 'valid')
    #         output = tf.layers.flatten(output)
    #         output = tf.layers.dense(output,squeeze_size,kernel_regularizer=self.regularizer,name="dense_1")
    #         output = tf.nn.relu(output)
    #         output = tf.layers.dense(output, output_num, kernel_regularizer=self.regularizer,name="dense_2")
    #         output = tf.nn.sigmoid(output)
    #         self.seblock_weight.append(output)
    #         output = tf.expand_dims(output, 1)
    #         output = tf.expand_dims(output, 1)
    #         output = input*output
    #         return output

    def get_train_step(self, learning_rate, ues_regularizer=False):
        self.loss = self.cross_entropy
        if(ues_regularizer):
            self.loss += tf.reduce_sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            )
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(self.loss)
        return train_step

    def get_variable(self,name):
        with tf.variable_scope("",reuse=True):
            return tf.get_variable(name=name)

    def load_weight(self,sess,saver,weight_saver_dir):
        try:
            saver.restore(sess, weight_saver_dir)
            return True
        except BaseException as e:
            print(e)
            sess.run(tf.global_variables_initializer())
            return False

