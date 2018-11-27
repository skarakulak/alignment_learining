#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:51:16 2018

@author: serkankarakulak
"""
import argparse
import multiprocessing as mp
from model import *

    
parser = argparse.ArgumentParser(description='Trains a model to recover the true signal, given noisy observations with cyclic shifts')
parser.add_argument('-E','--epochs', default=1, help='Number of epochs',required=True)
parser.add_argument('-v','--version', default='NOT_SPECIFIED', help='Version name of the model')
parser.add_argument('-d','--signal-dim', default=5, help='Dimension of the underlying signal')
parser.add_argument('-o','--n-obs', default=128, help='Number of observed signals')
parser.add_argument('-n','--sigma', default=2, help='Noise in the observed signals')
parser.add_argument('-b','--minibatch-size', default=64, help='Minibatch size during the training')
parser.add_argument('-t','--test-sample-size', default=64, help='Sample size of the test batches')
parser.add_argument('-e','--eval-n-times', default=16, help='Number of batches to be tested at evaluation')
parser.add_argument('-l','--learning-rate', default=0.00003, help='Initial learning rate')
parser.add_argument('-s','--skip-step', default=100, help='Frequency of test loss reports')
parser.add_argument('-i','--layer-invar-skip-step', default=100, help='Frequency of shift invariance repors of the layers')
parser.add_argument('-c','--num-of-processes', default=999999, help="Number of parallel processes to be used for data preparation (default value is 'threadnum - 1')")
parser.add_argument('-a','--eval-after', default=0, help='After which step to start generating test loss logs')

args = parser.parse_args()
threadNum = mp.cpu_count()-1 if args.num_of_processes == 999999 else args.num_of_processes


model = objGenNetwork(
        signalDim = int(args.signal_dim),
         nObservationsPerSignal = int(args.n_obs),
         noise = int(args.sigma),
         minibatchSize = int(args.minibatch_size),
         testSampleSize = int(args.test_sample_size),
         evalNTimes=(int(args.eval_n_times)),
         lr = float(args.learning_rate),
         training = True,
         skipStep = int(args.skip_step),
         layerInvarSkipStep = int(args.layer_invar_skip_step),
         nProcessesDataPrep=int(threadNum),
         vers=args.version,
         evalAfterStep=int(args.eval_after)
         )

model.build()

print('\nversion name: ' + model.vers +'\n')

model.train(n_epochs=int(args.epochs))
