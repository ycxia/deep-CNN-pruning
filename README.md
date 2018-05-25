# VGG16-filter-available
为了方便神经网络剪枝压缩的研究，实现了VGG16网络结构，能够对VGG16网络进行filter层面的剪枝。

## 文件说明
- VGG16Cifar10.py VGG16应用在cifar10数据集上的模型，13层卷积+2个全连接，中间加入bn和dropout
- VGG16SEBlock.py 带有seblock的VGG16
- ResNet20.py 残差网络原论文中提到的在cifar10上训练的模型结构。
- tensorReduce.py 用于张量裁剪，可以剪去卷积核某个输出通道，或者减少某个输入通道对应的卷积参数，从而进行剪枝


## 训练数据
dataset参数即数据集文件夹路径，目前的模型使用的都是cifar10-python数据集。

## 训练命令
选择GPU：CUDA_VISIBLE_DEVICES="4"
### 训练普通vgg16
python3 deep-CNN-pruning/main.py --epoch 100 --learning_rate 0.1 --batch_size 128 --checkpoint_dir=checkpoint/vgg16_cifar10/vgg16 --model_name=VGG16Cifar10

### 训练有seblock的vgg16
python3 deep-CNN-pruning/main.py --epoch 100 --learning_rate 0.1 --batch_size 128 --checkpoint_dir=checkpoint/vgg16_seblock/vgg16_seblock --model_name=VGG16SEBlock

### 训练ResNet20
python3 deep-CNN-pruning/main.py --epoch 100 --learning_rate 0.1 --batch_size 128 --l1_lambda=0.00001 --checkpoint_dir=checkpoint/ResNet20/ResNet20_slim --model_name=ResNet20

### 训练ResNet20SEBlock
python3 deep-CNN-pruning/main.py --epoch 100 --learning_rate 0.1 --batch_size 128 --checkpoint_dir=checkpoint/ResNet20SEBlock/ResNet20SEBlock --model_name=ResNet20SEBlock
