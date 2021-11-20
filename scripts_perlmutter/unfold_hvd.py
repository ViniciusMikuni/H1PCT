from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from pct import PCT
import time
import horovod.tensorflow.keras as hvd


def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss)


class Multifold():
    def __init__(self,nvars, niter,Q2,pct=False,version = 'Closure',verbose=1):
        # mc_gen, mc_reco, data,
        self.nvars = nvars
        self.niter=niter
        self.mc_gen = None
        self.mc_reco = None
        self.data=None
        self.version = version
        self.pct = pct
        self.Q2=Q2
        self.timing_log = 'time_keeper_{}'.format(self.version)
        if self.pct:
            self.timing_log +='_PCT'
        self.timing_file = open('time_keeper/{}.txt'.format(self.timing_log),'w')

    def Unfold(self):
        self.BATCH_SIZE=5000
        self.EPOCHS=1000
        self.weights_pull = np.ones(self.weights_mc.shape[0])
        self.weights_push = np.ones(self.weights_mc.shape[0])
        time_steps_1 = []
        time_steps_2 = []
        for i in range(self.niter):
            self.iter = i
            #self.CompileModel(max(1e-4/(2**i),1e-7))
            self.CompileModel(1e-4)
            
            if hvd.rank() ==0:
                self.log_string("ITERATION: {}".format(i + 1))
                start_time = time.time()
                
            self.RunStep1(i)
            
            if hvd.rank() ==0:
                self.log_string("Total time Step 1: {}".format(time.time()-start_time))
                time_steps_1.append(time.time()-start_time)
                start_time = time.time()

            self.RunStep2(i)
            if hvd.rank() ==0:
                self.log_string("Total time Step 2: {}".format(time.time()-start_time))
                time_steps_2.append(time.time()-start_time)
        if hvd.rank() ==0:
            self.log_string("Average time spent on Step 1: {}".format(np.average(time_steps_1)))
            self.log_string("Average time spent on Step 2: {}".format(np.average(time_steps_2)))
            
    def RunStep1(self,i):
        #Data versus reco MC reweighting
        print("RUNNING STEP 1")
        weights = np.concatenate((self.weights_push*self.weights_mc,self.weights_data ))        
        self.RunModel(np.concatenate((self.mc_reco, self.data)),np.concatenate((self.labels_mc, self.labels_data)),weights,np.concatenate((self.Q2['reco'], self.Q2['data'])),i,stepn=1)

        if self.pct:
            new_weights=self.reweight([self.mc_reco,self.Q2['reco']])
        else:
            new_weights=self.reweight(self.mc_reco)
        new_weights[self.not_pass_reco]=1.0
        self.weights_pull = self.weights_push *new_weights
        self.weights_pull = self.weights_pull/np.average(self.weights_pull)
    def RunStep2(self,i):
        #Gen to Gen reweighing
        print("RUNNING STEP 2")
        #self.weights_push*
        weights = np.concatenate((self.weights_mc, self.weights_pull*self.weights_mc))
        #weights = np.concatenate((self.weights_push, self.weights_pull))
        #weights = np.concatenate((np.ones(self.weights_mc.shape[0]), self.weights_pull))
        self.RunModel(np.concatenate((self.mc_gen, self.mc_gen)),np.concatenate((self.labels_mc, self.labels_gen)),weights,np.concatenate((self.Q2['gen'], self.Q2['gen'])),i,stepn=2)
        
        if self.pct:
            new_weights=self.reweight([self.mc_gen,self.Q2['gen']])
        else:
            new_weights=self.reweight(self.mc_gen)
        new_weights[self.not_pass_gen]=1.0
        #self.weights_push * 
        self.weights_push = new_weights
        self.weights_push = self.weights_push/np.average(self.weights_push)
    def RunModel(self,sample,labels,weights,Q2,iteration,stepn):
        
        X_train, X_test, Y_train, Y_test, w_train, w_test,q2_train,q2_test = train_test_split(sample, labels, weights,Q2,test_size=0.1)
        
        del sample
        del labels
        del weights
        del Q2
        
        Y_train = np.stack((Y_train, w_train), axis=1)
        Y_test = np.stack((Y_test, w_test), axis=1)
        if self.pct: #per particle
            mask_train = X_train[:,0,0]!=-10
            mask_test = X_test[:,0,0]!=-10
            train_data = tf.data.Dataset.from_tensor_slices((
                {'input_1':X_train[mask_train],
                 'input_2':q2_train[mask_train]},
                Y_train[mask_train])).batch(self.BATCH_SIZE)
            
            test_data = tf.data.Dataset.from_tensor_slices((
                {'input_1':X_test[mask_test],
                 'input_2':q2_test[mask_test]},
                Y_test[mask_test])).batch(self.BATCH_SIZE)

        else:
            mask_train = X_train[:,0]!=-10
            mask_test = X_test[:,0]!=-10
            # train_data = tf.data.Dataset.from_tensor_slices((X_train[mask_train], Y_train[mask_train])).shard(hvd.size(), hvd.rank()).repeat().batch(self.BATCH_SIZE)
            # test_data = tf.data.Dataset.from_tensor_slices((X_test[mask_test], Y_test[mask_test])).shard(hvd.size(), hvd.rank()).repeat().batch(self.BATCH_SIZE)

            train_data = tf.data.Dataset.from_tensor_slices((X_train[mask_train], Y_train[mask_train])).batch(self.BATCH_SIZE)
            test_data = tf.data.Dataset.from_tensor_slices((X_test[mask_test], Y_test[mask_test])).batch(self.BATCH_SIZE)

        
        #free memory after batching
        del X_train
        del X_test
        del Y_train
        del Y_test
        del w_train
        del w_test
        del q2_train
        del q2_test
        
        verbose = 1 if hvd.rank() == 0 else 0

        callbacks=[
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(initial_lr=self.hvd_lr, warmup_epochs=3, verbose=0),
            ReduceLROnPlateau(patience=5, verbose=0),
            EarlyStopping(patience=10,restore_best_weights=True)
        ]
        
        if hvd.rank() ==0:
            pass
            base_name = "Omnifold"
            log_name = 'logs'
            if self.pct:
                base_name+='_PCT'
                log_name+='_PCT'
                
            callbacks.append(ModelCheckpoint('../weights/{}_{}_iter{}_step{}.h5'.format(base_name,self.version,iteration,stepn),save_best_only=True,mode='auto',period=1,save_weights_only=True))
            #callbacks.append(TensorBoard(log_dir="logs/{}_{}_step{}".format(log_name,self.version,stepn)))
        
        hist =  self.model.fit(train_data,
                          epochs=self.EPOCHS,
                          #steps_per_epoch=int(NSAMPLES_TRAIN/(self.BATCH_SIZE*hvd.size())),
                          validation_data=test_data,
                          #validation_steps=int(NSAMPLES_TEST/(self.BATCH_SIZE*hvd.size())),
                          callbacks=callbacks,
                          verbose=verbose
        )


    def Preprocessing(self,weights_mc=None,weights_data=None):
        self.PrepareWeights(weights_mc,weights_data)
        self.PrepareInputs()
        self.PrepareModel()
        
    def PrepareWeights(self,weights_mc,weights_data):
        if self.pct:
            self.not_pass_reco = self.mc_reco[:,0,0]==-10
            self.not_pass_gen = self.mc_gen[:,0,0]==-10
        else:
            self.not_pass_reco = self.mc_reco[:,0]==-10
            self.not_pass_gen = self.mc_gen[:,0]==-10
            
        if weights_mc is None:
            self.weights_mc = np.ones(len(self.mc_reco))
        else:
            self.weights_mc = weights_mc
    
        if weights_data is None:
            self.weights_data = np.ones(len(self.data))
        else:
            self.weights_data =weights_data

        
    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc_reco))        
        self.labels_data = np.ones(len(self.data))
        self.labels_gen = np.ones(len(self.mc_gen))

        if not self.pct:
            scaler = StandardScaler()
            scaler.fit(self.mc_gen[self.mc_gen[:,0]!=-10])
            self.data[self.data[:,0]!=-10]=scaler.transform(self.data[self.data[:,0]!=-10])
            self.mc_reco[self.mc_reco[:,0]!=-10]=scaler.transform(self.mc_reco[self.mc_reco[:,0]!=-10])
            self.mc_gen[self.mc_gen[:,0]!=-10]=scaler.transform(self.mc_gen[self.mc_gen[:,0]!=-10])

    def CompileModel(self,lr):
        self.hvd_lr = lr*hvd.size()
        opt = tensorflow.keras.optimizers.Adam(learning_rate=self.hvd_lr)
        opt = hvd.DistributedOptimizer(
            opt, backward_passes_per_step=1, average_aggregated_gradients=True)
        
        self.model.compile(loss=weighted_binary_crossentropy,
                           optimizer=opt,
                           experimental_run_tf_function=False
        )
        
            
    def PrepareModel(self):
        if self.pct:
            inputs,input_q2,outputs = PCT(20,4,nheads=4)
            self.model = Model(inputs=[inputs,input_q2], outputs=outputs)
        else:          
            inputs = Input((self.nvars, ))
            layer = Dense(50, activation='relu')(inputs)
            layer = Dense(100, activation='relu')(layer)
            layer = Dense(50, activation='relu')(layer)
            outputs = Dense(1, activation='sigmoid')(layer)
            self.model = Model(inputs=inputs, outputs=outputs)
        
    def reweight(self,events):

        f = np.nan_to_num(self.model.predict(events, batch_size=5000),posinf=0,neginf=0)
        weights = f / (1. - f)
        weights = weights[:,0]
        return np.squeeze(np.nan_to_num(weights,posinf=1))

        
    def log_string(self,out_str):
        self.timing_file.write(out_str+'\n')
        self.timing_file.flush()
        print(out_str)
