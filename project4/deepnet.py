# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:11:01 2019

@author: Jerry Xing
"""
import numpy as np
from get_transition_func import get_transition_func
from forward_pass import forward_pass
from compute_loss import compute_loss
from backprop import backprop
def deepnet(Ws, xTr, yTr, wst, transname = 'sigmoid'):
    
    entry = np.cumsum(wst[0:-1] * wst[1:] + wst[0:-1]) # entry points into weights
    if Ws.size == 0:
        Ws = np.random.randn(entry[-1], 1) / 2
    
    W = []
    e = 0

    for i in range(len(entry)):
        W.append(np.reshape(Ws[e:entry[i]], [wst[i], wst[i+1]+1]))
        e = entry[i]#???
        
    trans_func, trans_func_der = get_transition_func(transname)
    
    aas, zzs = forward_pass(W, xTr, trans_func)
    
    if len(yTr) == 0:
        loss = zzs[0]
        return loss
    
    loss = compute_loss(zzs, yTr)
    gradientList = backprop(W, aas,zzs, yTr, trans_func_der)
    
    #%% reformat the gradient from a cell-array of matrices to one vector
    gradient = np.zeros((entry[-1], 1))
    e = 0
    for i in range(len(entry)):
        gradient[e:entry[i]] = np.reshape(gradientList[i], (entry[i] - e, 1))
        e = entry[i]
        
    return loss, gradient