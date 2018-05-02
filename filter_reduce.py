import tensorflow as tf

#去除输出第index个通道的相关卷积核
def reduceFilterOutput(input, index, sess, realLen):
    shape = input.shape
    left = tf.slice(input, [0,0,0,0], [shape[0],shape[1],shape[2],index])
    right = tf.slice(input, [0,0,0,index+1], [shape[0],shape[1],shape[2],realLen-index-1])
    ret = tf.concat([left, right], 3)
    sess.run(tf.assign(input, ret, validate_shape=False))
    return ret

#去除处理第index个输入通道的相关卷积核
def reduceFilterInput(input, index, sess, realLen):
    shape = input.shape
    left = tf.slice(input, [0,0,0,0], [shape[0],shape[1],index,shape[3]])
    right = tf.slice(input, [0,0,index+1,0], [shape[0],shape[1],realLen-index-1,shape[3]])
    ret = tf.concat([left, right], 2)
    sess.run(tf.assign(input, ret, validate_shape=False))
    return ret

def reduceFilter(input, sess, target_index, target_dim, shape):
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
    right = tf.slice(input, rightStart, rightSize)
    ret = tf.concat([left, right], target_dim)
    sess.run(tf.assign(input, ret, validate_shape=False))
    shape[target_dim] = shape[target_dim]-1
    print(ret)