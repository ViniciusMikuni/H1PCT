import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, Conv1D, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
import sys


def MLP(nvars):
    inputs = Input((nvars, ))
    layer = Dense(50, activation='relu')(inputs)
    layer = Dense(100, activation='relu')(layer)
    #layer = Dropout(0.5)(layer)
    layer = Dense(50, activation='relu')(layer)
    #layer = Dropout(0.5)(layer)
    outputs = Dense(1, activation='sigmoid')(layer)
    return inputs,outputs
