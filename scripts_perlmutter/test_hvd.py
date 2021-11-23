import horovod.tensorflow.keras as hvd
import tensorflow as tf
import numpy as np
#from mpi4py import MPI


hvd.init()

assert hvd.mpi_threads_supported()
gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# if gpus:
#     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

#rank = MPI.COMM_WORLD.rank
print(80*'#')
print(hvd.size(), hvd.rank(), hvd.local_size(), hvd.local_rank())
print(80*'#')


def SampleSharding(size):
    shards = np.linspace(0,size,hvd.size()+1,dtype=int)
    begin = shards[hvd.rank()]
    end = shards[hvd.rank()+1]
    return begin,end


print(SampleSharding(10e6))

