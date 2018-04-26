
import os
from glob import glob
import util
from util import Cifar10Dataset
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from VGG16SEBlock import VGG16SEBlock

model = VGG16SEBlock(0.0001)
model.build_model()
model.build_model()




