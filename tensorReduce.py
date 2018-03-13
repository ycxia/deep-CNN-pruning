import tensorflow as tf

#去除输出第index个通道的相关卷积核
def reduceFilterOutput(input, index):
    shape = input.shape
    left = tf.slice(input, [0,0,0,0], [shape[0],shape[1],shape[2],index])
    right = tf.slice(input, [0,0,0,index+1], [shape[0],shape[1],shape[2],shape[3]-index-1])
    return tf.concat([left, right], 3)

#去除处理第index个输入通道的相关卷积核
def reduceFilterInput(input, index):
    shape = input.shape
    left = tf.slice(input, [0,0,0,0], [shape[0],shape[1],index,shape[3]])
    right = tf.slice(input, [0,0,index+1,0], [shape[0],shape[1],shape[2]-index-1,shape[3]])
    return tf.concat([left, right], 2)