# import tensorflow as tf
import numpy as np
from sa import SA
from common import *

model = SA(shape=[48,48,1],no_classes=7)
model.build_model()
data = np.zeros((7,100,48,48,1))
val = np.ones((20,48,48,1))
labels_val = one_hot(np.ones((20)), 7)
model.train(data, val, labels_val, period_save_model=5)