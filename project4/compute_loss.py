# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:03:45 2019

@author: remus
"""
import numpy as np
def compute_loss(zzs, yTr):
#% function [loss] = compute_loss(zs, yTr)
#%
#% INPUT:
#% zzs output of forward pass (list of numpy ndarray)
#% yTr 1xn numpy ndarray (each entry is a label)
#%
#% OUTPUTS:
#% 
#% loss = the total loss obtained with w on xTr and yTr, or the prediction of yTr is not passed on
#%

    delta = zzs[0] - yTr
    n = np.shape(yTr)[1]
    loss = 0
    #% INSERT CODE HERE:
    loss = np.linalg.norm(delta)**2/(2*n)
    return loss