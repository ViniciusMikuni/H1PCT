import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import tensorflow as tf

from sklearn.metrics import roc_curve, auc
from tensorflow.keras.losses import binary_crossentropy, mse
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
import sys
sys.path.append('../')
from shared.pct import PCT


import horovod.tensorflow.keras as hvd

hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return max(lr * tf.math.exp(-epoch/5.),1e-6)


parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--folder", type="string", default="/global/cfs/cdirs/m1759/vmikuni/", help="Folder containing input files")
parser.add_option("--batch", type=int, default=32, help="Batch size")
(flags, args) = parser.parse_args()

files = ['train_ttbar.h5','test_ttbar.h5']

train = {
    'X':h5.File(os.path.join(flags.folder, 'train_ttbar.h5'),'r')['data'][:],
    'y':h5.File(os.path.join(flags.folder, 'train_ttbar.h5'),'r')['pid'][:]
}

test = {
    'X':h5.File(os.path.join(flags.folder, 'test_ttbar.h5'),'r')['data'][:],
    'y':h5.File(os.path.join(flags.folder, 'test_ttbar.h5'),'r')['pid'][:]
}

train_data = tf.data.Dataset.from_tensor_slices((train['X'],train['y'])).shuffle(train['X'].shape[0]).shard(hvd.size(), hvd.rank()).batch(flags.batch)
test_data = tf.data.Dataset.from_tensor_slices((test['X'],test['y'])).batch(flags.batch)


inputs,outputs = PCT(train['X'].shape[1],train['X'].shape[2],nheads=4)

callbacks=[
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=1e-4*hvd.size(), warmup_epochs=3, verbose=0),
    EarlyStopping(patience=10,restore_best_weights=True),
    ReduceLROnPlateau(patience=3, verbose=1),
    #tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
]


opt = tf.keras.optimizers.Adam(1e-5*np.sqrt(hvd.size()))
opt = hvd.DistributedOptimizer(
    opt, backward_passes_per_step=1, average_aggregated_gradients=True)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy',
              optimizer=opt,              
              metrics=['accuracy'],
              experimental_run_tf_function=False
)

verbose = 1 if hvd.rank() == 0 else 0
hist =  model.fit(train_data,
                  epochs=100,
                  #steps_per_epoch=int(train['X'].shape[0]/(1.0*flags.batch*hvd.size())),
                  validation_data=test_data,
                  callbacks=callbacks,
                  verbose=verbose,
)

