# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:57:03 2019

@author: remus
"""
import scipy.io as sio
import numpy as np
import time
import pickle

from preprocess import preprocess
from initweights import initweights
from grdescent import grdescent
from deepnet import deepnet

def bostontest():
    bostonData = sio.loadmat('./boston.mat')
    # boston_secretData = sio.loadmat('./boston_secret.mat')
    boston_secretData = sio.loadmat('./boston.mat')
    
    with open('best_parameters.pickle', 'rb') as f:
        best_parameters = pickle.load(f)
    
    xTr = bostonData['xTr']
    yTr = bostonData['yTr']
    # xTe_secret = boston_secretData['xTe_secret']
    # yTe_secret = boston_secretData['yTe_secret']
    xTe_secret = bostonData['xTr']
    yTe_secret = bostonData['yTr']
        
    TRANSNAME = best_parameters['TRANSNAME']
    ROUNDS = best_parameters['ROUNDS']
    ITER = best_parameters['ITER']
    STEPSIZE = best_parameters['STEPSIZE']
    wst = best_parameters['wst'].astype(np.int)
    
    runs = 5;
    times = np.zeros((1,runs));
    errors = np.zeros((1,runs));
    
    for r in range(runs):
        t1 = time.time()
        xTr, xTe_secret, _, _ = preprocess(xTr, xTe_secret)
        
        # Do training
        w = initweights(wst)
        err = []
        
        f = lambda w: deepnet(w, xTr, yTr, wst, TRANSNAME)
        for i in range(ROUNDS):
            w = grdescent(f, w, STEPSIZE, ITER, 1e-8)
            predTes=deepnet(w,xTe_secret,[],wst,TRANSNAME)            
            err.append(np.sqrt(np.mean((predTes-yTe_secret)**2)));
        times[0,r] = time.time() - t1
        errors[0,r] = err[-1]
    
    final_time = np.min(times)
    final_err = np.min(errors)
    
    print(times)
    print(errors)
    
    if final_time > 500:
    	time_score = 0
    elif final_time > 200:
    	time_score = 1
    elif final_time > 100:
      	time_score = 2
    elif final_time > 50:
      	time_score = 3
    elif final_time > 30:
      	time_score = 4
    elif final_time > 20:
      	time_score = 5
    elif final_time > 15:
      	time_score = 6
    elif final_time > 10:
      	time_score = 7
    elif final_time > 5:
      	time_score = 8
    elif final_time > 2:
        time_score = 9
    else:
      	time_score = 10    
    
    if final_err > 10:
    	error_score = 0
    elif final_err > 4:
      	error_score = 1
    elif final_err > 3.7:
      	error_score = 2
    elif final_err > 3.5:
      	error_score = 3
    elif final_err > 3.3:
      	error_score = 4
    elif final_err > 3.2:
      	error_score = 5
    elif final_err > 3.1:
      	error_score = 6
    elif final_err > 3.0:
      	error_score = 7
    elif final_err > 2.9:
      	error_score = 8
    elif final_err > 2.8:
      	error_score = 9
    else:
      	error_score = 10
    
    return time_score, error_score
time, err = bostontest()
print(time)
print(err)
