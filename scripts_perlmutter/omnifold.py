from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import sys, os
import horovod.tensorflow.keras as hvd
import horovod.tensorflow
sys.path.append('../')
from shared.pct import PCT
from shared.dense import MLP
import json, yaml
from datetime import datetime

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss)

def label_smoothing(y_true,alpha=0):
    '''Regularization through binary label smoothing'''
    new_label = y_true*(1-alpha) + (1-y_true)*alpha
    return new_label

def Scaler(file,var_list):
    '''Standardize a set of features stored in var_list accessible through file'''
    
    mean = [np.average(file[var],weights=file[var_list[0]][:]>0) for var in var_list]
    variance = [np.average((file[var]-mean[iv])**2, weights=file[var_list[0]][:]>0) for iv, var in enumerate(var_list)]
    return np.array(mean),np.sqrt(variance)


def LoadJson(file_name):
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))


class Multifold():
    def __init__(self,nevts,nstrap=0,mode='hybrid',version = 'Closure',config_file='config_omnifold.json',verbose=1):
        self.mode = mode
        assert self.mode in ['PCT','hybrid','standard'], "ERROR: Omnifold mode not implemented"
        
        self.version = version
        self.nevts = nevts
        self.verbose = verbose
        self.log_file =  open('log_{}.txt'.format(self.version),'w')
        self.opt = LoadJson(config_file)
        self.niter = self.opt['General']['NITER']
        self.nstrap=nstrap
        self.mc_gen = None
        self.mc_reco = None
        self.global_vars = None
        self.data=None

        # tf.random.set_seed(19874*self.nstrap)
        # np.random.seed(19874*self.nstrap)
        self.weights_folder = '../weights'
        if self.nstrap>0:
            self.weights_folder = '../weights_strap'
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)
            
    def Unfold(self):
        self.BATCH_SIZE=self.opt['General']['BATCH_SIZE']
        self.EPOCHS=self.opt['General']['EPOCHS']
        self.CompileModel(float(self.opt['General']['LR']))
                                        
        self.weights_pull = np.ones(self.weights_mc.shape[0])
        self.weights_push = np.ones(self.weights_mc.shape[0])
        
        
        patience = 0
        max_patience = 5
        min_distance = 1e6
        for i in range(self.niter):
            #self.CompileModel(max(1e-4/(2**i),1e-6))
            #self.CompileModel(max(float(self.opt['General']['LR'])/(2**i),1e-7))
            print("ITERATION: {}".format(i + 1))            
            self.RunStep1(i)        
            old_weights=self.weights_push
            self.RunStep2(i)            
            if hvd.rank() == 0:
                patience,min_distance = self.CompareDistance(patience,min_distance,old_weights,self.weights_push)
                #if patience >= max_patience:
                #    break
                

    def RunStep1(self,i):
        '''Data versus reco MC reweighting'''
        print("RUNNING STEP 1")

        if self.mode != 'standard':
           global_vars =  np.concatenate((self.global_vars['reco'], self.global_vars['data']))
        else:
            global_vars = []
            
        self.RunModel(
            np.concatenate((self.mc_reco, self.data)),
            np.concatenate((self.labels_mc, self.labels_data)),
            np.concatenate((self.weights_push*self.weights_mc,self.weights_data )),
            i,self.model1,stepn=1,
            global_vars=global_vars
        )
        
        if self.mode == 'standard':
            new_weights=self.reweight(self.mc_reco,self.model1)
        else:
            new_weights=self.reweight([self.mc_reco,self.global_vars['reco']],self.model1)
            
        new_weights[self.not_pass_reco]=1.0
        self.weights_pull = self.weights_push *new_weights
        self.weights_pull = self.weights_pull/np.average(self.weights_pull)

    def RunStep2(self,i):
        '''Gen to Gen reweighing'''        
        print("RUNNING STEP 2")

        if self.mode == 'PCT':
            global_vars = np.concatenate((self.global_vars['gen'], self.global_vars['gen']))
        else:
            global_vars = []
            
        self.RunModel(
            np.concatenate((self.mc_gen, self.mc_gen)),
            np.concatenate((self.labels_mc, self.labels_gen)),
            np.concatenate((np.ones(self.weights_mc.shape), self.weights_pull)),
            i,self.model2,stepn=2,
            global_vars=global_vars
        )

        if self.mode == 'PCT':
            new_weights=self.reweight([self.mc_gen,self.global_vars['gen']],self.model2)
        else:
            new_weights=self.reweight(self.mc_gen,self.model2)


        new_weights[self.not_pass_gen]=1.0
        self.weights_push = new_weights
        self.weights_push = self.weights_push/np.average(self.weights_push)

    def RunModel(self,sample,labels,weights,iteration,model,stepn,global_vars=None):
        
        if (self.mode == 'PCT') or  (self.mode == 'hybrid' and stepn==1): #Only reco is per particle if hybrid
            mask = sample[:,0,0]!=-10
            if stepn==1:
                data = tf.data.Dataset.from_tensor_slices((
                    {'input_1':sample[mask],
                     'input_2':global_vars[mask]}, 
                    np.stack((labels[mask],weights[mask]),axis=1))
                ).shuffle(np.sum(mask))
            else:
                data = tf.data.Dataset.from_tensor_slices((
                    {'input_3':sample[mask],
                     'input_4':global_vars[mask]}, 
                    np.stack((labels[mask],weights[mask]),axis=1))
                ).shuffle(np.sum(mask))
                
        else:
            mask = sample[:,0]!=-10
            if self.verbose: print("SHUFFLE BUFFER",np.sum(mask))
            data = tf.data.Dataset.from_tensor_slices((
                sample[mask],
                np.stack((labels[mask],weights[mask]),axis=1))
            ).cache().shuffle(np.sum(mask))

            

        #Fix same number of training events between ranks
        NTRAIN,NTEST = self.GetNtrainNtest(stepn)        
        test_data = data.take(NTEST).repeat().batch(self.BATCH_SIZE)
        train_data = data.skip(NTEST).repeat().batch(self.BATCH_SIZE)

        if self.verbose:
            print(80*'#')
            print("Train events used: {}, total number of train events: {}, percentage: {}".format(NTRAIN,np.sum(mask)*0.8, np.sum(mask)*0.8/NTRAIN))
            print(80*'#')

        verbose = 2 if hvd.rank() == 0 else 0
        
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(
                initial_lr=self.hvd_lr, warmup_epochs=self.opt['General']['NWARMUP'],
                verbose=verbose),
            ReduceLROnPlateau(patience=5, min_lr=1e-7,verbose=verbose),
            EarlyStopping(patience=self.opt['General']['NPATIENCE'],restore_best_weights=True)
        ]
        
        base_name = "Omnifold_{}".format(self.mode)
        
        if hvd.rank() ==0:
            if self.nstrap>0:                
                callbacks.append(
                    ModelCheckpoint('{}/{}_{}_iter{}_step{}_strap{}.h5'.format(self.weights_folder,base_name,self.version,iteration,stepn,self.nstrap),
                                    save_best_only=True,mode='auto',period=1,save_weights_only=True))
            else:
                callbacks.append(
                    ModelCheckpoint('{}/{}_{}_iter{}_step{}.h5'.format(self.weights_folder,base_name,self.version,iteration,stepn),
                                    save_best_only=True,mode='auto',period=1,save_weights_only=True))
                
        _ =  model.fit(
            train_data,
            epochs=self.EPOCHS,
            steps_per_epoch=int(NTRAIN/self.BATCH_SIZE),
            validation_data=test_data,
            validation_steps=int(NTEST/self.BATCH_SIZE),
            verbose=verbose,
            callbacks=callbacks)




    def Preprocessing(self,weights_mc=None,weights_data=None):
        self.PrepareWeights(weights_mc,weights_data)
        self.PrepareInputs()
        self.PrepareModel()

    def PrepareWeights(self,weights_mc,weights_data):
        
        if self.mode =='PCT':
            self.not_pass_reco = self.mc_reco[:,0,0]==-10
            self.not_pass_gen = self.mc_gen[:,0,0]==-10
        elif self.mode =='hybrid':
            self.not_pass_reco = self.mc_reco[:,0,0]==-10
            self.not_pass_gen = self.mc_gen[:,0]==-10
        else:
            self.not_pass_reco = self.mc_reco[:,0]==-10
            self.not_pass_gen = self.mc_gen[:,0]==-10

        
        if weights_mc is None:
            self.weights_mc = np.ones(self.mc_reco.shape[0])
        else:
            self.weights_mc = weights_mc

        if weights_data is None:
            self.weights_data = np.ones(self.data.shape[0])
        else:
            self.weights_data =weights_data


    def CompileModel(self,lr):
        #self.hvd_lr = lr*np.sqrt(hvd.size())
        self.hvd_lr = lr*hvd.size()
        opt = tensorflow.keras.optimizers.Adam(learning_rate=self.hvd_lr)
        opt = hvd.DistributedOptimizer(
            opt, average_aggregated_gradients=True)

        self.model1.compile(loss=weighted_binary_crossentropy,
                            optimizer=opt,experimental_run_tf_function=False)

        self.model2.compile(loss=weighted_binary_crossentropy,
                            optimizer=opt,experimental_run_tf_function=False)


    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc_reco))
        self.labels_data = np.ones(len(self.data))
        self.labels_gen = np.ones(len(self.mc_gen))

        # self.labels_mc = label_smoothing(np.zeros(len(self.mc_reco)),0.01)
        # self.labels_data = label_smoothing(np.ones(len(self.data)),0.01)
        # self.labels_gen = label_smoothing(np.ones(len(self.mc_gen)),0.01)


    def PrepareModel(self):
        if self.mode == 'PCT':
            nglobal = self.global_vars['reco'].shape[1]
            input1,input_global1,outputs1 = PCT(self.opt['PCT']['NPOINT'],4,self.opt['PCT']['NHEAD'],nglobal)
            input2,input_global2,outputs2 = PCT(self.opt['PCT']['NPOINT'],4,self.opt['PCT']['NHEAD'],nglobal)
            inputs1 = [input1,input_global1]
            inputs2 = [input2,input_global2]
                        
        elif self.mode == 'standard':
            nvars = self.mc_gen.shape[1]
            inputs1,outputs1 = MLP(nvars,self.opt['MLP']['NTRIAL'])
            inputs2,outputs2 = MLP(nvars,self.opt['MLP']['NTRIAL'])
                        
        elif self.mode == 'hybrid':
            nglobal = self.global_vars['reco'].shape[1]
            nvars = self.mc_gen.shape[1]
            input1,input_global1,outputs1 = PCT(self.opt['PCT']['NPOINT'],4,self.opt['PCT']['NHEAD'],nglobal)
            inputs1 = [input1,input_global1]
            inputs2,outputs2 = MLP(nvars,self.opt['MLP']['NTRIAL'])
            
        self.model1 = Model(inputs=inputs1, outputs=outputs1)
        self.model2 = Model(inputs=inputs2, outputs=outputs2)


    def GetNtrainNtest(self,stepn):
        if stepn ==1:
            #about 20% acceptance for reco events
            NTRAIN=int(0.2*0.8*self.nevts/hvd.size())
            NTEST=int(0.2*0.2*self.nevts/hvd.size())                        
        else:
            NTRAIN=int(0.8*self.nevts/hvd.size())
            NTEST=int(0.2*self.nevts/hvd.size())                        

        return NTRAIN,NTEST

    def reweight(self,events,model):
        f = np.nan_to_num(model.predict(events, batch_size=10000),posinf=1,neginf=0)
        weights = f / (1. - f)
        #weights = np.clip(weights,0,10)
        weights = weights[:,0]
        return np.squeeze(np.nan_to_num(weights,posinf=1))

    def CompareDistance(self,patience,min_distance,weights1,weights2):
        distance = np.mean(
            (np.sort(weights1) - np.sort(weights2))**2)
                
        print(80*'#')
        self.log_string("Distance between weights: {}".format(distance))
        print(80*'#')
                
        if distance<min_distance:
            min_distance = distance
            patience = 0
        else:
            print(80*'#')
            print("Distance increased! before {} now {}".format(min_distance,distance))
            print(80*'#')
            patience+=1
        return patience, min_distance
        
    def log_string(self,out_str):
        self.log_file.write(out_str+'\n')
        self.log_file.flush()
        print(out_str)
