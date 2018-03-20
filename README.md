# VGG16-filter-available
为了方便神经网络剪枝压缩的研究，实现了VGG16网络结构，能够对VGG16网络进行filter层面的剪枝。

## 文件说明
- vgg16.py 网络模型类，使用build_model方法进行模型构建，每层的卷积核可以在成员属性中获得
- tensorReduce.py 用于张量裁剪，可以剪去卷积核某个输出通道，或者减少某个输入通道对应的卷积参数，从而进行剪枝

## 训练命令
python main.py --epoch 20 --learning_rate 0.02 --batch_size 16 --dataset cifar-10-batches-py --testset_size 32
