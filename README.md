# VGG16-filter-available
为了方便神经网络剪枝压缩的研究，实现了VGG16网络结构，能够对VGG16网络进行filter层面的剪枝。

## 文件说明
- VGG16Cifar10.py VGG16应用在cifar10数据集上的模型，使用build_model方法进行模型构建，每层的卷积核可以在成员属性中获得
- tensorReduce.py 用于张量裁剪，可以剪去卷积核某个输出通道，或者减少某个输入通道对应的卷积参数，从而进行剪枝

## 训练数据
训练数据需要存放在项目下的data文件夹下（自行创建），dataset参数即数据集文件夹的名称。VGG16Cifar10使用的是cifar10-python数据集。

## 训练命令
### 训练普通vgg16
!python VGG16-filter-available/main.py --epoch 500 --learning_rate 0.00006 --batch_size 64 --dataset cifar-10-batches-py --testset_size 1000 --l2_lambda=0.0007 --checkpoint_dir=drive/weights/vgg16_cifar10/vgg16_cifar_epoch --model_name=VGG16Cifar10

### 训练有seblock的vgg16
!python VGG16-filter-available/main.py --epoch 500 --learning_rate 0.00006 --batch_size 64 --dataset cifar-10-batches-py --testset_size 1000 --l2_lambda=0.0007 --checkpoint_dir=drive/weights/vgg16_seblock --model_name=VGG16SEBlock

