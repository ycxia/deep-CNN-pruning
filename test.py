
import os
from glob import glob
import util
from util import Cifar10Dataset
import pickle
import matplotlib.pyplot as plt

import numpy as np
arr2 = np.array([
    [1,2,3],
    [4,5,6]
])
class_count = np.zeros(shape=(10))
class_count[5] += 1
print(class_count[5]==0)





