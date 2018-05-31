
import tensorflow as tf
import numpy as np
import filter_reduce as fr



def vgg_prune(sess, data_set , model, moudle_name,moudle_name_next, compress_rate, channel_index):
    # train_step = model.get_train_step(learning_rate, ues_regularizer)

    # load_result = model.load_weight(sess, saver, checkpoint_dir)
    # if load_result == True:
    #     print("Checkpoint load success!")
    # else:
    #     print("No checkpoint file,weight inited!")
    filters_dict = {}
    shape_dict = {}
    with tf.variable_scope(moudle_name,reuse=True):
        filters_dict["pre_filter"] = tf.get_variable("conv/kernel")
        shape_dict["pre_filter"] = tensor_shape_to_int(filters_dict["pre_filter"].shape)
        filters_dict["bn_gm"] = tf.get_variable("bn/gamma")
        shape_dict["bn_gm"] = tensor_shape_to_int(filters_dict["bn_gm"].shape)
        filters_dict["bn_mean"] = tf.get_variable("bn/moving_mean")
        shape_dict["bn_mean"] = tensor_shape_to_int(filters_dict["bn_mean"].shape)
        filters_dict["bn_var"] = tf.get_variable("bn/moving_variance")
        shape_dict["bn_var"] = tensor_shape_to_int(filters_dict["bn_var"].shape)
        filters_dict["bn_beta"] = tf.get_variable("bn/beta")
        shape_dict["bn_beta"] = tensor_shape_to_int(filters_dict["bn_beta"].shape)
        filters_dict["se_filter1"] = tf.get_variable("seblock/dense_1/kernel")
        shape_dict["se_filter1"] = tensor_shape_to_int(filters_dict["se_filter1"].shape)
        filters_dict["se_filter2"] = tf.get_variable("seblock/dense_2/kernel")
        shape_dict["se_filter2"] = tensor_shape_to_int(filters_dict["se_filter2"].shape)
    with tf.variable_scope(moudle_name_next, reuse=True):
        filters_dict["next_filter"] = tf.get_variable("conv/kernel")
        shape_dict["next_filter"] = tensor_shape_to_int(filters_dict["next_filter"].shape)

    seblock_output = model.seblock_output[channel_index]
    seblock_output = sess.run(seblock_output,feed_dict={model.x: data_set.prune_x,model.isTrain: False})
    print(str(type(seblock_output)))
    seblock_prune(sess, filters_dict, seblock_output, compress_rate, shape_dict)

def tensor_shape_to_int(shape):
    ret = []
    for i in shape:
        ret.append(int(i))
    return ret

# def thinet_channel_select(sess,input_channel,filters):
#     input_channel_num = input_channel.shape[1]
#     input_channel_size = input_channel.shape[2]
#     filter_num = filters.shape[3]
#     input_channel = np.pad(input_channel,((0,0), (1,1),(1,1),(0,0)), 'constant')
#     print(input_channel.shape)
#     prune_sample = np.array()

def seblock_channel_select(channel_weight,compress_rate):
    """
    :param channel_weight:numpy数组
        seblock的输出
    :param compress_rate:float
        压缩率
    :return numpy数组
        返回需要剪枝的通道index，index从大到小（方便剪枝）:
    """
    # channel_weight = np.reshape(channel_weight,(-1,channel_weight.shape[3]))

    channel_weight = np.mean(channel_weight,axis=(0))
    prune_num = int(channel_weight.shape[0]*compress_rate)
    index = channel_weight.argsort()[0:prune_num]
    index.sort()
    index = index[::-1]
    return index

def seblock_prune(sess, filters_dict, seblock_output, compress_rate, shape_dict):
    prune_indexs = seblock_channel_select(seblock_output, compress_rate)
    pre_filter = filters_dict["pre_filter"]
    bn_gm = filters_dict["bn_gm"]
    bn_mean = filters_dict["bn_mean"]
    bn_var = filters_dict["bn_var"]
    bn_beta = filters_dict["bn_beta"]
    next_filter = filters_dict["next_filter"]
    se_dense1 = filters_dict["se_filter1"]
    se_dense2 = filters_dict["se_filter2"]
    for i in prune_indexs:
        fr.reduce_conv_filter_output(sess,pre_filter,i,shape_dict["pre_filter"])
        fr.reduce_bn_filter(sess,bn_gm,i,shape_dict["bn_gm"])
        fr.reduce_bn_filter(sess, bn_mean, i, shape_dict["bn_mean"])
        fr.reduce_bn_filter(sess, bn_var, i, shape_dict["bn_var"])
        fr.reduce_bn_filter(sess, bn_beta, i, shape_dict["bn_beta"])
        fr.reduce_conv_filter_input(sess,next_filter,i,shape_dict["next_filter"])
        fr.reduce_dense_filter_input(sess,se_dense1,i,shape_dict["se_filter1"])
        fr.reduce_dense_filter_output(sess,se_dense2,i,shape_dict["se_filter2"])


