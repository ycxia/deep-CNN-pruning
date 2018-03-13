import vgg16
import tensorflow as tf

def run():
    batch_size = 64
    train_data_size = None
    epoch_num = None

    test_x, test_label = None

    vgg = vgg16()
    vgg.build_model()
    train_step = vgg.get_train_step(0.02)

    print("Training starts!")
    batch_num = batch_size/train_data_size
    with tf.Session() as sess:
        for epoch in range(epoch_num):
            for i in range(batch_num):
                batch_x ,batch_label= None #batch_x和batch_label需要在这里进行赋值
                sess.run(train_step,feed_dict={vgg.x:batch_x,vgg.y_:batch_label})
                if i%100==0:
                    loss = sess.run(vgg.cross_entropy, feed_dict={vgg.x: test_x, vgg.y_: test_label})
                    print(str(i) + "/" + str(batch_num) + " batch: loss is " + str(loss))
            loss,acc = sess.run([vgg.cross_entropy,vgg.accaury], feed_dict={vgg.x: test_x, vgg.y_: test_label})
            print(str(epoch) + " epoch: loss is " + str(loss) + ",accuary is " + str(acc))
    print("Training end!")

