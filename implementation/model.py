#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:51:16 2018

@author: serkankarakulak
"""

import numpy as np 
import tensorflow as tf
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import os
from src import *

def dense(x, scope, num_h, n_x):
    """
    Standard affine layer
    
    scope = name tf variable scope
    num_h = number of hidden units
    num_x = number of input units
    """
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [n_x, num_h], initializer=tf.random_normal_initializer(stddev=0.04))
        b = tf.get_variable('b', [num_h], initializer=tf.constant_initializer(0))
        return tf.matmul(x, w)+b

def lrelu(x, alpha, name='lrelu'):
    return(tf.identity(tf.nn.relu(x) - alpha * tf.nn.relu(-x), name=name) )
    
def conv(x, scope, filter_h,filter_w, n_kernel, stride_h=1,stride_w=1, padding='SAME'):
    """
    Convolutional layer
    
    scope        = name tf variable scope
    filter_h     = height of the receptive field
    filter_w     = width of the receptive field
    n_kernel     = # of kernels
    stride_h     = stride height
    stride_w     = stride width
    """
    with tf.variable_scope(scope, reuse=False):
        n_x = x.get_shape().as_list()[-1]
        w = tf.get_variable('w',
                            [filter_h, filter_w, n_x, n_kernel],
                            initializer=tf.random_normal_initializer(stddev=0.04))
        b = tf.get_variable('b', [n_kernel], initializer=tf.constant_initializer(0))
        return tf.nn.convolution(x, w, padding=padding, strides=[stride_h, stride_w])+b    


def bnorm(X,isTraining,scope='batch_norm',axis=-1):
    """
    Batch normalization layer
    
    X          = input
    isTraining = True during training, False otherwise.
    axis       = axis for normalization
    scope      = name tf variable scope
    
    """
    return(tf.layers.batch_normalization(
        inputs=X,
        axis=axis, 
        training = isTraining,
        name=scope
    ))
    
def convObservations(X,batch_size,n_images,isTraining=False, reuse=False):
    """
    Makes convolutions over noisy signals with cyclic shifts.
    X.size = (batch_size,nObservations, signalDim,1 ). Returns the average 
    value of the encodings of the observations
    """

    with tf.variable_scope('convObservations', reuse=reuse):
        conv_0 =  conv(X, 'conv0',1,5,32,1,1)
        lrelu0 = lrelu(conv_0, 0.1,'lrelu0')  # 5x32
        bnorm0 = bnorm(lrelu0,isTraining,'bnorm_1d_0')
        conv_1 =  conv(bnorm0, 'conv1',1,3,64,1,1)
        lrelu1 = lrelu(conv_1, 0.1,'lrelu1')  # 5x64
        bnorm1 = bnorm(lrelu1,isTraining,'bnorm_1d_1')
        conv_2 =  conv(bnorm1, 'conv2',1,3,128,1,2)
        lrelu2 = lrelu(conv_2, 0.1,'lrelu2')  # 3x128
        bnorm2 = bnorm(lrelu2,isTraining,'bnorm_1d_2')
        conv_3 =  conv(bnorm2, 'conv3',1,3,256,1,2)
        lrelu3 = lrelu(conv_3, 0.1,'lrelu3')  # 2x256
        bnorm3 = bnorm(lrelu3,isTraining,'bnorm_1d_3')
        conv_4 =  conv(bnorm3, 'conv4',1,2,512,1,2)
        lrelu4 = lrelu(conv_4, 0.1,'lrelu4')  # 1x512
        bnorm4 = bnorm(lrelu4,isTraining,'bnorm_1d_4')

        fc_0 = tf.layers.dense(bnorm4, 512, name='fc_0')
        fl_0 = lrelu(fc_0,0.1,'flrelu_0')
        bnorm_fc_0 = bnorm(fl_0,isTraining,'bnorm_1d_fc0')
        fc_1 = tf.layers.dense(bnorm_fc_0, 512, name='fc_1')
        fl_1 = lrelu(fc_1,0.1, name='flrelu_1')

        h = tf.reduce_mean(fl_1, [1,2])

        return(
            conv_0,
            lrelu0,
            bnorm0,
            conv_1,
            lrelu1,
            bnorm1,
            conv_2,
            lrelu2,
            bnorm2,
            conv_3,
            lrelu3,
            bnorm3,
            conv_4,
            lrelu4,
            bnorm4,
            fc_0,
            fl_0,
            bnorm_fc_0,
            fc_1,
            fl_1,
            h
            )





    
def decodeSignal(X, batch_size, enc_dim = 512, isTraining=False, reuse=False):
    """
    Takes encoding produced by the observations as input and
    generates a the underlying true signal
    """
    with tf.variable_scope('decodeSignal', reuse=reuse):
        h = lrelu(dense(X, 'hz0', num_h=256,n_x=enc_dim), 0.1)
        hz = tf.layers.dense(h, 5, name='z0')
        h = bnorm(h,isTraining,'bnorm_hz0')
        h = lrelu(dense(h, 'hz1', num_h=128,n_x=256), 0.1)
        hz = tf.add(hz, tf.layers.dense(h, 5, name='z1'))
        h = bnorm(h,isTraining,'bnorm_hz1')
        h = lrelu(dense(h, 'hz2', num_h=64,n_x=128), 0.1)
        hz = tf.add(hz, tf.layers.dense(h, 5, name='z2'))
        h = bnorm(h,isTraining,'bnorm_hz2')
        h = lrelu(dense(h, 'hz3', num_h=32,n_x=64), 0.1)
        hz = tf.add(hz, tf.layers.dense(h, 5, name='z3'))
        h = bnorm(h,isTraining,'bnorm_hz3')
        h = lrelu(dense(h, 'hz4', num_h=16,n_x=32), 0.1)
        hz = tf.add(hz, tf.layers.dense(h, 5, name='z4'))
        h = bnorm(h,isTraining,'bnorm_hz4')
        h = lrelu(dense(h, 'hz5', num_h=8,n_x=16), 0.1)
        hz = tf.add(hz, tf.layers.dense(h, 5, name='z5'))
        h = lrelu(dense(h, 'hz6', num_h=5,n_x=8), 0.1)
        hz = tf.add(hz, tf.layers.dense(h, 5, name='z6'))
        
        return(hz)    
    

class objGenNetwork(object):
    """
    Implementation of the model
    """
    def __init__(self,
                 signalDim = 5,
                 nObservationsPerSignal = 64,
                 noise = 2,
                 minibatchSize = 64,
                 testSampleSize = 1000,
                 lr = 0.001,
                 training = True,
                 skipStep = 1,
                 layerInvarSkipStep=1,
                 nProcessesDataPrep=4,
                 vers='NOT_SPECIFIED',
                 evalAfterStep=0,
                 evalNTimes=1
                ):
        self.signalDim = signalDim
        self.nObservationsPerSignal = nObservationsPerSignal
        self.noise = noise
        self.minibatchSize = minibatchSize
        self.testSampleSize = testSampleSize
        self.lr = lr
        self.isTraining = training
        self.skipStep = skipStep
        self.layerInvarSkipStep = layerInvarSkipStep
        self.nProcessesDataPrep = nProcessesDataPrep
        if (vers=='NOT_SPECIFIED'):
            self.vers = str(signalDim)+'D'+'_sigma_'+str(noise) +'_obs_' + str(nObservationsPerSignal )
        else:
            self.vers = vers
        self.logFile = 'log_'+ self.vers +'.txt'
        self.evalAfterStep = evalAfterStep
        self.evalNTimes = evalNTimes
        
        self.gstep = tf.Variable(0, 
                                 dtype=tf.int32, 
                                 trainable=False,
                                 name='global_step')
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def data_generator(self,trainingBatch=True):

        """
        Generates batches of random signals and their noisy 
        observations with cyclic shifts. |Signal| = dim(Signal)
        """
        minibatchSize = self.minibatchSize
        testSampleSize = self.testSampleSize
        signalDim = self.signalDim
        noise = self.noise
        nObservationsPerSignal=self.nObservationsPerSignal
        
        
        batches = minibatchSize if trainingBatch else testSampleSize

        
        poolData = mp.Pool(processes= self.nProcessesDataPrep)
        results = poolData.starmap(genSignal,[(
            signalDim,
            nObservationsPerSignal,
            noise)] * batches )
        poolData.close(); poolData.join()

        batch_x = np.expand_dims(
            np.array([k[1] for k in results],dtype='float32'),
            axis=3
        )
        batch_y = np.array([k[0] for k in results],dtype='float32')
        

        if trainingBatch:
            self.train_x_new = batch_x
            self.train_y_new = batch_y
        else:
            self.test_x_new = batch_x
            self.test_y_new = batch_y
        
    def genDataForVariantStudy(self,trainingBatch=False):
        """
        Generates batches of random signals and for each signal
        one noisy observation in all possible rotations
        |Signal| = dim(Signal)
        """
        testSampleSize = self.testSampleSize
        signalDim = self.signalDim
        noise = self.noise        
        
        batches = testSampleSize

        poolData = mp.Pool(processes= self.nProcessesDataPrep)
        results = poolData.starmap(genSignalAndAllShifts,[(
            signalDim,
            noise)] * batches )
        poolData.close(); poolData.join()

        batch_x = np.expand_dims(
            np.array([k[1] for k in results],dtype='float32'),
            axis=3
        )

        self.layerInv_batch_x_new = batch_x


        
    def inference(self):
        self.internal_layers = convObservations(
            self.x_ph, 
            self.minibatchSize,
            self.nObservationsPerSignal,
            isTraining=self.isTraining
            )

        self.preds = decodeSignal(
            self.internal_layers[-1], 
            self.minibatchSize,
            isTraining=self.isTraining
            )

    def loss(self):
        """
        Defines loss function
        We use mean squared loss over the predicted and the true signal
        under the best fitting cyclic shift.
        """
        # 
        with tf.name_scope('loss'):
            tiled_preds = tf.tile(tf.expand_dims(self.preds, 1),[1,self.signalDim,1])
            entropy = tf.squared_difference(self.y_ph,tiled_preds)
            entropy = tf.reduce_sum(entropy, axis = 2)
            entropy = tf.reduce_min(entropy,axis = 1)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        """
        Optimization op
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, 
                                                                   global_step=self.gstep)

    def additionalEvalMetrics(self):
        """
        Takes the most accurate rotation of the true object
        as reference and calculates the average accuracy of
        the occupancy grid of the object
        """
        with tf.name_scope('prediction_eval'):
            tiledPreds = tf.tile(tf.expand_dims(self.preds, 1),
                                 [1,self.signalDim,1])
            MAE = tf.reduce_sum(tf.abs(tiledPreds - self.y_ph),
                                axis = 2)
            MAE = tf.reduce_min(MAE,axis=1)
            self.MAE = tf.reduce_mean(MAE,name="MAE")


    def layerEval(self, sess,step, evalNTimes):        
        self.isTraining = False

        # numpy arrays that would store the norm of the shift invariant
        # and shift variant parts of the layer activations. 
        layerInvarNorms = np.zeros(
            (
                self.testSampleSize * evalNTimes, 
                len(list(self.internal_layers)) - 1
                )
            )

        layerVariantPartNorms = np.copy(layerInvarNorms)

        for i in range(evalNTimes):
            layerInv_batch_x = self.layerInv_batch_x_new
            
            pool = ThreadPool(processes=1)
            # preparation of the next batch
            async_result = pool.apply_async(self.genDataForVariantStudy,(False,))
            # activations of the layers
            obsLayers = sess.run(
                [list(self.internal_layers)[:-1]],
                feed_dict={self.x_ph: layerInv_batch_x}
                )
            for ind,l in enumerate(obsLayers[0]):
                n_s = i * self.testSampleSize        # for indexing
                n_end = (i+1) * self.testSampleSize  # for indexing
                if(len(l.shape)==4):
	                # mean activation of the layer. tiled for comparing different shifts
	                invariant = np.tile(l.mean(axis=1),(1,self.signalDim,1)).reshape(l.shape)
	                layerInvarNorms[n_s:n_end,ind]=np.sqrt(np.sum(np.square(invariant),axis=(1,2,3)) )
	                # norm of the variant part of the activation. 
	                variantPart = np.sqrt(np.sum(np.square(l - invariant),axis=(1,2,3)) )
	                layerVariantPartNorms[n_s:n_end,ind] = variantPart
                elif(len(l.shape)==3):
	                # mean activation of the layer. tiled for comparing different shifts
	                invariant = np.tile(l.mean(axis=1),(1,self.signalDim)).reshape(l.shape)
	                layerInvarNorms[n_s:n_end,ind]=np.sqrt(np.sum(np.square(invariant),axis=(1,2)) )
	                # norm of the variant part of the activation. 
	                variantPart = np.sqrt(np.sum(np.square(l - invariant),axis=(1,2)) )
	                layerVariantPartNorms[n_s:n_end,ind] = variantPart
            pool.close()
            pool.join()

        self.layersAverageInvarNorm = np.mean(layerInvarNorms,axis=0)
        self.layersAverageVariantPartNorm = np.mean(layerVariantPartNorms,axis=0)
        self.layersVarInvarRatios = np.mean(layerInvarNorms/layerVariantPartNorms, axis=0)

        if (self.logFile!=False):
            if(not os.path.isfile('norm'+ self.logFile)):
                with open('norm'+ self.logFile,'a') as lgfile:   
                    lgfile.write('\t'+ '\t'.join( [k.name.split('/')[1] for k in list(self.internal_layers)[:-1] ]) +'\n' )
                    lgfile.write('\t'.join([str(step)]+['{:.5}'.format(k) for k in list(self.layersVarInvarRatios)]) +'\n' )

            else:
                with open('norm'+ self.logFile,'a') as lgfile:
                    lgfile.write('\t'.join([str(step)] +['{:.5}'.format(k) for k in list(self.layersVarInvarRatios)]) +'\n')
                

    def testEval(self, sess, step,evalNTimes):
        self.isTraining = False
        l2_arr = np.zeros((evalNTimes,))
        mae_arr = np.zeros((evalNTimes,))
        for i in range(evalNTimes):
            self.test_x = np.copy(self.test_x_new)
            self.test_y = np.copy(self.test_y_new)
            
            pool = ThreadPool(processes=1)
            async_result = pool.apply_async(self.data_generator,(False,))
            mae_arr[i], l2_arr[i] = sess.run(
                    [self.MAE,self.loss],
                     feed_dict={self.x_ph: self.test_x,
                                self.y_ph: self.test_y}
                     )
            pool.close()
            pool.join()


        if (self.logFile==False):
            print('test MAE at step {0:.6}: {1:.6} '.format(step,mae_arr.mean()))
            print('test loss at step {0:.6}: {1:.6} '.format(step,l2_arr.mean()))
        else:
            with open(self.logFile,'a') as lgfile:
                lgfile.write('{0}\t{1:.6}\t{2:.6}\n'.format(step,l2_arr.mean(),mae_arr.mean()))
            

    def summary(self):
        """
        Summary for TensorBoard
        """
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('MAE', self.MAE)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build(self):
        """
        Builds the computation graph
        """
        self.x_ph = tf.placeholder(tf.float32, [None, 
                                                None,
                                                self.signalDim,
                                                1]) 
        self.y_ph = tf.placeholder(tf.float32, [None,
                                                self.signalDim,
                                                self.signalDim])
        self.data_generator()
        self.data_generator(trainingBatch=False)
        self.genDataForVariantStudy()
        self.inference()
        self.loss()
        self.optimize()
        self.additionalEvalMetrics()
        self.summary()
    
    def train_one_epoch(self, sess, saver, writer, epoch, step):
#        start_time = time.time()
        self.isTraining = True
        _, l, summaries = sess.run([self.opt, 
                                    self.loss,
                                    self.summary_op],
                                   feed_dict={self.x_ph: self.train_x_new,
                                              self.y_ph: self.train_y_new})
        writer.add_summary(summaries, global_step=step)
        #if (step + 1) % self.skipStep == 0:
        #    print('training Loss at step {0}: {1}'.format(step, l))
        step += 1
        saver.save(sess, 'checkpoints/cryoem_'+self.vers+'/cpoint', global_step=step)
#        print('Average loss at epoch {0}: {1}'.format(epoch, l))
#        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def train(self, n_epochs):
        """
        Calls the training ops and prepares the training data
        for the next batch in a parallel process.
        """
        safe_mkdir('checkpoints')
        safe_mkdir('checkpoints/cryoem_'+self.vers)
        writer = tf.summary.FileWriter('./graphs/cryoem_'+self.vers, tf.get_default_graph())

        tVars = tf.trainable_variables()
        defGraph = tf.get_default_graph()

        for v in defGraph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES): 
            if (('bnorm_' in v.name) and
                ('/Adam' not in v.name) and
                ('Adagrad' not in v.name) and
                (v not in tVars )):
                tVars.append(v)
                
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(var_list= tVars)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/cryoem_'+self.vers+'/cpoint'))

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                self.train_x = np.copy(self.train_x_new)
                self.train_y = np.copy(self.train_y_new)
                
                pool = ThreadPool(processes=1)
                async_result = pool.apply_async(self.data_generator,())
                step = self.train_one_epoch(sess, saver, writer, epoch, step)
                pool.close()
                pool.join()

                if ( ((step + 1) % self.skipStep == 0) and (step>self.evalAfterStep ) ) :
                    self.testEval(sess,step, self.evalNTimes)
                if ( ((step + 1) % self.layerInvarSkipStep == 0) and (step>self.evalAfterStep ) ) :
                    self.layerEval(sess,step,self.evalNTimes)

        writer.close()
        self.isTraining = False
