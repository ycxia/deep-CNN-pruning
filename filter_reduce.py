import tensorflow as tf

#去除输出第index个通道的相关卷积核
def reduce_conv_filter_output(sess, input, index, shape):
    reduceFilter(sess, input, index, 3, shape)

#去除处理第index个输入通道的相关卷积核
def reduce_conv_filter_input(sess, input, index, shape):
    reduceFilter(sess, input, index, 2, shape)

def reduce_bn_filter(sess, input, index, shape):
    reduceFilter(sess, input, index, 0, shape)

def reduce_dense_filter_output(sess, input, index, shape):
    reduceFilter(sess, input, index, 1, shape)

def reduce_dense_filter_input(sess, input, index, shape):
    reduceFilter(sess, input, index, 0, shape)

def reduceFilter(sess, input,  target_index, target_dim, shape):
    leftSize = []
    rightSize = []
    left_start = []
    rightStart = []
    for (num,i) in zip(shape,range(len(shape))):
        if(i==target_dim):
            leftSize.append(target_index)
            rightSize.append(num-1-target_index)
            rightStart.append(target_index+1)
        else:
            leftSize.append(num)
            rightSize.append(num)
            rightStart.append(0)
        left_start.append(0)
    left = tf.slice(input, left_start, leftSize)
    print(rightStart)
    print(rightSize)
    right = tf.slice(input, rightStart, rightSize)
    ret = tf.concat([left, right], target_dim)
    sess.run(tf.assign(input, ret, validate_shape=False))
    shape[target_dim] = shape[target_dim]-1