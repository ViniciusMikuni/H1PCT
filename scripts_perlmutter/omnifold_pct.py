from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
import time
import horovod.tensorflow.keras as hvd
import sys
sys.path.append('../')
from shared.pct import PCT


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
    def __init__(self,niter,global_vars,nglobal,nevts,pct=False,version = 'Closure',nhead = 1,verbose=1):
        # mc_gen, mc_reco, data,
        self.niter=niter
        self.mc_gen = None
        self.mc_reco = None
        self.data=None
        self.version = version
        self.pct = pct
        self.global_vars=global_vars
        self.nglobal = nglobal
        self.timing_log = 'time_keeper_{}'.format(self.version)
        self.nhead=nhead
        self.nevts = nevts
        
        if self.pct:
            self.timing_log +='_PCT'
        self.timing_file = open('time_keeper/{}.txt'.format(self.timing_log),'w')

    def Unfold(self):
        self.BATCH_SIZE=128
        self.EPOCHS=10
        self.weights_pull = np.ones(self.weights_mc.shape[0])
        self.weights_push = np.ones(self.weights_mc.shape[0])
        time_steps_1 = []
        time_steps_2 = []
        
        self.CompileModel(1e-5)
        min_distance=1e6
        
        for i in range(self.niter):
            self.iter = i
            #self.CompileModel(max(1e-4/(2**i),1e-7))
            if hvd.rank() ==0:
                self.log_string("ITERATION: {}".format(i + 1))
                start_time = time.time()
                
            self.RunStep1(i)
            old_weights=self.weights_push
            if hvd.rank() ==0:
                self.log_string("Total time Step 1: {}".format(time.time()-start_time))
                time_steps_1.append(time.time()-start_time)
                start_time = time.time()

            self.RunStep2(i)
            if hvd.rank() ==0:
                self.log_string("Total time Step 2: {}".format(time.time()-start_time))
                time_steps_2.append(time.time()-start_time)
                
                distance = np.mean((np.sort(old_weights) - np.sort(self.weights_push))**2)
                self.log_string(80*'#')
                self.log_string("Distance between weights: {}".format(distance))
                self.log_string(80*'#')
                if distance<min_distance:
                    min_distance = distance                
                else:
                    self.log_string(80*'#')
                    self.log_string("Distance increased! before {} now {}".format(min_distance,distance))
                    self.log_string(80*'#')
                
                    break


                
        if hvd.rank() ==0:
            self.log_string("Average time spent on Step 1: {}".format(np.average(time_steps_1)))
            self.log_string("Average time spent on Step 2: {}".format(np.average(time_steps_2)))
            
    def RunStep1(self,i):
        '''Data versus reco MC reweighting'''
        print("RUNNING STEP 1")
        weights = np.concatenate((self.weights_push*self.weights_mc,self.weights_data ))        
        self.RunModel(
            np.concatenate((self.mc_reco, self.data)),
            np.concatenate((self.labels_mc, self.labels_data)),
            weights,i,stepn=1,
            global_vars=np.concatenate((self.global_vars['reco'], self.global_vars['data'])))
        if self.pct:
            new_weights=self.reweight([self.mc_reco,self.global_vars['reco']])
        else:
            new_weights=self.reweight(self.mc_reco)
            
        new_weights[self.not_pass_reco]=1.0
        self.weights_pull = self.weights_push *new_weights
        self.weights_pull = self.weights_pull/np.average(self.weights_pull)
    def RunStep2(self,i):
        '''Gen to Gen reweighing'''
        print("RUNNING STEP 2")

        weights = np.concatenate((np.ones(self.weights_mc.shape[0]), self.weights_pull))
        self.RunModel(
            np.concatenate((self.mc_gen, self.mc_gen)),
            np.concatenate((self.labels_mc, self.labels_gen)),
            weights,i,stepn=2,
            global_vars=np.concatenate((self.global_vars['gen'], self.global_vars['gen'])))
        
        if self.pct:
            new_weights=self.reweight([self.mc_gen,self.global_vars['gen']])
        else:
            new_weights=self.reweight(self.mc_gen)
        new_weights[self.not_pass_gen]=1.0
        #self.weights_push * 
        self.weights_push = new_weights
        self.weights_push = self.weights_push/np.average(self.weights_push)
        
    def RunModel(self,sample,labels,weights,iteration,stepn,global_vars):

        if self.pct: #Only reco is per particle
            mask = sample[:,0,0]!=-10
            data = tf.data.Dataset.from_tensor_slices((
                {'input_1':sample[mask],
                 'input_2':global_vars[mask]}, 
                np.stack((labels[mask],weights[mask]),axis=1))
            ).shuffle(labels.shape[0]).repeat()
            
        else:
            mask = sample[:,0]!=-10
            data = tf.data.Dataset.from_tensor_slices((
                sample[mask],
                np.stack((labels[mask],weights[mask]),axis=1))
            ).shuffle(labels.shape[0]).repeat()

        
        #Fix same number of training events between ranks
        if stepn ==1:
            #about 20% acceptance for reco events
            NTRAIN=int(0.2*0.8*self.nevts/hvd.size())
            NTEST=int(0.2*0.2*self.nevts/hvd.size())                        
        else:
            NTRAIN=int(0.8*self.nevts/hvd.size())
            NTEST=int(0.2*self.nevts/hvd.size())                        
        
        test_data = data.take(NTEST).batch(self.BATCH_SIZE)
        train_data = data.skip(NTEST).batch(self.BATCH_SIZE)
        
        verbose = 1 if hvd.rank() == 0 else 0

        callbacks=[
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(
                initial_lr=self.hvd_lr, warmup_epochs=5, verbose=verbose),
            #ReduceLROnPlateau(patience=5, verbose=verbose),
            EarlyStopping(patience=10,restore_best_weights=True)
        ]
        
        if hvd.rank() ==0:
            pass
            base_name = "Omnifold"
            log_name = 'logs'
            if self.pct:
                base_name+='_PCT'
                log_name+='_PCT'
                
            callbacks.append(ModelCheckpoint('../weights/{}_{}_perlmutter_iter{}_step{}.h5'.format(base_name,self.version,iteration,stepn),save_best_only=True,mode='auto',period=1,save_weights_only=True))
            #callbacks.append(TensorBoard(log_dir="logs/{}_{}_step{}".format(log_name,self.version,stepn)))
        
        hist =  self.model.fit(
            train_data,
            epochs=self.EPOCHS,
            steps_per_epoch=int(NTRAIN/self.BATCH_SIZE),
            validation_data=test_data,
            validation_steps=int(NTEST/self.BATCH_SIZE),
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
            mean = np.mean(self.mc_gen[self.mc_gen[:,0]!=-10],0)
            std = np.std(self.mc_gen[self.mc_gen[:,0]!=-10],0)
 
            self.data[self.data[:,0]!=-10] = (self.data[self.data[:,0]!=-10]-mean)/std
            self.mc_reco[self.mc_reco[:,0]!=-10]= (self.mc_reco[self.mc_reco[:,0]!=-10]-mean)/std        
            self.mc_gen[self.mc_gen[:,0]!=-10]=(self.mc_gen[self.mc_gen[:,0]!=-10]-mean)/std
        else:
            mask_reco = self.global_vars['reco'][:,0]>0
            mask_data = self.global_vars['data'][:,0]>0
            mask_gen = self.global_vars['gen'][:,0]>0
            
            mean = np.mean(self.global_vars['gen'][mask_reco],0)
            std = np.std(self.global_vars['gen'][mask_reco],0)
            
            self.global_vars['reco'][mask_reco] = (self.global_vars['reco'][mask_reco]-mean)/std
            self.global_vars['data'][mask_data] = (self.global_vars['data'][mask_data]-mean)/std
            self.global_vars['gen'][mask_gen] = (self.global_vars['gen'][mask_gen]-mean)/std
            

    def CompileModel(self,lr):
        self.hvd_lr = lr*np.sqrt(hvd.size())
        opt = tensorflow.keras.optimizers.Adam(learning_rate=self.hvd_lr)
        opt = hvd.DistributedOptimizer(
            opt)
        #average_aggregated_gradients=True
        self.model.compile(loss=weighted_binary_crossentropy,
                           optimizer=opt,
                           experimental_run_tf_function=False
        )
        
            
    def PrepareModel(self):
        if self.pct:
            inputs,input_global,outputs = PCT(20,4,nheads=self.nhead,nglobal=self.nglobal,)
            self.model = Model(inputs=[inputs,input_global], outputs=outputs)
        else:
            nvars = self.mc_gen.shape[1]
            inputs = Input((nvars, ))
            layer = Dense(50, activation='relu')(inputs)
            layer = Dense(100, activation='relu')(layer)
            layer = Dropout(0.5)(layer)
            layer = Dense(50, activation='relu')(layer)
            layer = Dropout(0.5)(layer)
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
