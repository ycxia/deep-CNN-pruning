import tensorflow as tf
import numpy as np
import tensorReduce as tr
x = tf.Variable(np.random.rand(10,3,3,4), dtype=np.float32)
filter = tf.Variable(np.random.rand(2, 2, 4, 2), dtype=np.float32)
y = tf.nn.conv2d(x,filter,[1,1,1,1],'SAME')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))
    reduced = tr.reduceFilterOutput(filter,0)
    reduced = tf.assign(filter,reduced,validate_shape=False)
    sess.run(reduced)