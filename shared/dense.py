import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
import sys



# def MLP(nvars,NTRIALS=10):
#     inputs = Input((nvars, ))
#     inputs_trials = tf.expand_dims(inputs,1)
#     inputs_trials = tf.tile(inputs_trials,[1,NTRIALS,1])
    
#     layer = Conv1D(50, kernel_size = 1, strides=1,
#                    activation='relu')(inputs_trials)
    
#     layer = Conv1D(nvars, kernel_size = 1, strides=1,
#                    activation='relu')(layer)

#     layer = Conv1D(50, kernel_size = 1, strides=1,
#                    activation='relu')(layer+inputs_trials)
    
#     layer = Conv1D(1, kernel_size = 1, strides=1,
#                    activation=None)(layer)
    
#     outputs = tf.reduce_mean(layer,1) #Average over trials
#     outputs =tf.keras.activations.sigmoid(outputs)

#     return inputs,outputs


def MLP(nvars,NTRIALS=10):
    inputs = Input((nvars, ))
    net_trials = []
    for _ in range(NTRIALS):            
        layer = Dense(50,activation='relu')(inputs)    
        layer1 = Dense(50, activation='relu')(layer)
        layer = Dense(50,activation='relu')(layer+layer1)    
        layer = Dense(1,activation='sigmoid')(layer)
        net_trials.append(layer)
        
    outputs = tf.reduce_mean(net_trials,0) #Average over trials
    return inputs,outputs
