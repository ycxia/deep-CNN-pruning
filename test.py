
import os
from glob import glob
import util
from util import Cifar10Dataset
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

t = tf.placeholder(tf.float64,[None,3,3,2])

print(t)



