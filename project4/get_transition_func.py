# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:19:32 2019

@author: Jerry Xing
"""
import numpy as np
def get_transition_func(transType):
#%	Given the type, gets a specific transition function
#%   INPUT
#%   type 'sigmoid', 'tanh', 'ReLU'
#%   OUTPUT
#%   trans_func transition function (function)
#%   trans_func_der derivative of the transition function (function)
    if transType.lower() == 'sigmoid':
        trans_func = lambda z: 1 / (1+np.exp(-z))
        trans_func_der = lambda z: trans_func(z)*(1-trans_func(z))
    elif transType.lower() == 'relu2':
        trans_func = lambda z: 0.5 * (np.maximum(z, 0)**2)
        trans_func_der = lambda z: (z>=0) * z
    elif transType.lower() == 'tanh':
        trans_func = lambda z: np.tanh(z)
        trans_func_der = lambda z: 1 - np.tanh(z)**2
    elif transType.lower() == 'relu':
        trans_func = lambda z: np.maximum(z, 0)
        trans_func_der = lambda z: z >= 0
    else:
        raise ValueError('Unsupported transition function type: ' + transType)
    
    return trans_func, trans_func_der