
import os
from glob import glob
import util
import pickle
import matplotlib.pyplot as plt

test_file = glob(os.path.join("data/cifar-10-batches-py", 'test*'))
test_x,test_label = util.data_read(test_file)
print("Test set shape:{}".format(test_x.shape))




