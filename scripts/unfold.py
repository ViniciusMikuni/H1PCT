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
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from pct import PCT

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
    def __init__(self,nvars,niter,Q2,pct=False,version = 'Closure',verbose=1):
        self.nvars = nvars
        self.niter=niter
        self.mc_gen = None
        self.mc_reco = None
        self.data=None
        self.version = version
        self.pct = pct
        self.Q2 = Q2



    def Unfold(self):
        self.BATCH_SIZE=1000
        self.EPOCHS=1000
        self.weights_pull = np.ones(self.weights_mc.shape[0])
        self.weights_push = np.ones(self.weights_mc.shape[0])


        for i in range(self.niter):
            self.iter = i
            self.CompileModel(1e-4/(2**i))
            print("ITERATION: {}".format(i + 1))
            self.RunStep1(i)
            self.RunStep2(i)

    def RunStep1(self,i):
        #Data versus reco MC reweighting
        print("RUNNING STEP 1")
        weights = np.concatenate((self.weights_push*self.weights_mc,self.weights_data ))
        self.RunModel(np.concatenate((self.mc_reco, self.data)),np.concatenate((self.labels_mc, self.labels_data)),weights,i,self.model1,stepn=1,Q2=np.concatenate((self.Q2['reco'], self.Q2['data'])))
        if self.pct:
            new_weights=self.reweight([self.mc_reco,self.Q2['reco']],self.model1)
        else:
            new_weights=self.reweight(self.mc_reco,self.model1)
        new_weights[self.not_pass_reco]=1.0

        self.weights_pull = self.weights_push *new_weights
        self.weights_pull = self.weights_pull/np.average(self.weights_pull)

    def RunStep2(self,i):
        #Gen to Gen reweighing
        print("RUNNING STEP 2")
        #self.weights_push*
        #weights = np.concatenate((self.weights_mc, self.weights_pull*self.weights_mc))
        #weights = np.concatenate((self.weights_push, self.weights_pull))
        weights = np.concatenate((np.ones(self.weights_mc.shape[0]), self.weights_pull))
        self.RunModel(np.concatenate((self.mc_gen, self.mc_gen)),np.concatenate((self.labels_mc, self.labels_gen)),weights,i,self.model2,stepn=2)
        
        new_weights=self.reweight(self.mc_gen,self.model2)
        new_weights[self.not_pass_gen]=1.0
        #self.weights_push * 
        self.weights_push = new_weights/np.average(new_weights)

    def RunModel(self,sample,labels,weights,iteration,model,stepn,Q2=None):

        if Q2 is not None:
            X_train, X_test, Y_train, Y_test, w_train, w_test, q2_train, q2_test = train_test_split(sample, labels, weights,Q2)
        else:
            X_train, X_test, Y_train, Y_test, w_train, w_test = train_test_split(sample, labels, weights)       
        
        Y_train = np.stack((Y_train, w_train), axis=1)
        Y_test = np.stack((Y_test, w_test), axis=1)

        del sample
        del labels
        del weights


        if self.pct and stepn==1: #Only reco is per particle
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

            train_data = tf.data.Dataset.from_tensor_slices((X_train[mask_train], Y_train[mask_train])).batch(self.BATCH_SIZE)
            test_data = tf.data.Dataset.from_tensor_slices((X_test[mask_test], Y_test[mask_test])).batch(self.BATCH_SIZE)

        del X_train
        del X_test
        del Y_train
        del Y_test
        del w_train
        del w_test

        base_name = "Omnifold"
        if self.pct:
            base_name+='_PCT'

        base_name = "Omnifold"
        if self.pct:
            base_name+='_PCT'

        callbacks = [
            ModelCheckpoint('weights/{}_{}_iter{}_step{}.h5'.format(base_name,self.version,iteration,stepn),save_best_only=True,mode='auto',period=1,save_weights_only=True),
            EarlyStopping(patience=10,restore_best_weights=True)
        ]


        hist =  model.fit(train_data,
                          epochs=self.EPOCHS,
                          validation_data=test_data,
                          callbacks=callbacks)
        return hist



    def Preprocessing(self,weights_mc=None,weights_data=None):
        self.PrepareWeights(weights_mc,weights_data)
        self.PrepareInputs()
        self.PrepareModel()

    def PrepareWeights(self,weights_mc,weights_data):
        if self.pct:
            self.not_pass_reco = self.mc_reco[:,0,0]==-10
        else:
            self.not_pass_reco = self.mc_reco[:,0]==-10

        self.not_pass_gen = self.mc_gen[:,0]==-10

        if weights_mc is None:
            self.weights_mc = np.ones(len(mc_reco))
        else:
            self.weights_mc = weights_mc

        if weights_data is None:
            self.weights_data = np.ones(len(data))
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
        opt = tensorflow.keras.optimizers.Adam(learning_rate=lr)

        self.model1.compile(loss=weighted_binary_crossentropy,
                            optimizer=opt)

        self.model2.compile(loss=weighted_binary_crossentropy,
                            optimizer=opt)


    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc_reco))        
        self.labels_data = np.ones(len(self.data))
        self.labels_gen = np.ones(len(self.mc_gen))

        
        scaler = StandardScaler()
        scaler.fit(self.mc_gen[self.mc_gen[:,0]!=-10])

        if not self.pct:
            self.data[self.data[:,0]!=-10]=scaler.transform(self.data[self.data[:,0]!=-10])
            self.mc_reco[self.mc_reco[:,0]!=-10]=scaler.transform(self.mc_reco[self.mc_reco[:,0]!=-10])        
        self.mc_gen[self.mc_gen[:,0]!=-10]=scaler.transform(self.mc_gen[self.mc_gen[:,0]!=-10])
        
                

    def PrepareModel(self):
        if self.pct:
            inputs,input_q2,outputs = PCT(20,4)
            self.model1 = Model(inputs=[inputs,input_q2], outputs=outputs)
        else:          
            inputs = Input((self.nvars, ))
            layer = Dense(50, activation='relu')(inputs)
            layer = Dense(100, activation='relu')(layer)
            layer = Dense(50, activation='relu')(layer)
            outputs = Dense(1, activation='sigmoid')(layer)
            self.model1 = Model(inputs=inputs, outputs=outputs)

        inputs2 = Input((self.nvars, ))
        layer = Dense(50, activation='relu')(inputs2)
        layer = Dense(100, activation='relu')(layer)
        layer = Dense(50, activation='relu')(layer)
        outputs2 = Dense(1, activation='sigmoid')(layer)



        self.model2 = Model(inputs=inputs2, outputs=outputs2)          

                
        

    def reweight(self,events,model):

        f = np.nan_to_num(model.predict(events, batch_size=5000),posinf=0,neginf=0)
        weights = f / (1. - f)
        weights = weights[:,0]
        return np.squeeze(np.nan_to_num(weights,posinf=1))

        
