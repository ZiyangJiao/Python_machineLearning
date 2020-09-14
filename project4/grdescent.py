# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:03:09 2019

@author: Jerry Xing
"""
import numpy as np
def grdescent(func, w0, stepsize, maxiter, tolerance = 1e-2):
#% function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
#%
#% INPUT:
#% func function to minimize
#% w0 = initial weight vector 
#% stepsize = initial gradient descent stepsize 
#% tolerance = if norm(gradient)<tolerance, it quits
#%
#% OUTPUTS:
#% 
#% w = final weight vector
#%
    w = w0
    ## << Insert your solution here
    for i in range(maxiter):
        tmp = stepsize*func(w)[1]
        if tolerance > np.linalg.norm(w - tmp):
            break
        while func(w)[0] < func(w-tmp)[0]:
            stepsize = stepsize * 0.5
            tmp = stepsize*func(w)[1]
        w = w - tmp
        stepsize = stepsize * 1.01
    return w