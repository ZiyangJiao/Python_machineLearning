# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:04:15 2019

@author: Jerry Xing
"""
import numpy as np
def checkgrad(f, X, e, *kwargs):

#% checkgrad checks the derivatives in a function, by comparing them to finite
#% differences approximations. The partial derivatives and the approximation
#% are printed and the norm of the diffrence divided by the norm of the sum is
#% returned as an indication of accuracy.
#%
#% usage: checkgrad(f, X, e, P1, P2, ...)
#%
#% where X is the argument and e is the small perturbation used for the finite
#% differences. and the P1, P2, ... are optional additional parameters which
#% get passed to f. The function f should be of the type 
#%
#% [fX, dfX] = f(X, P1, P2, ...)
#%
#% where fX is the function value and dfX is a vector of partial derivatives.
#%
#% Carl Edward Rasmussen, 2001-08-01.

    y, dy = f(X, *kwargs)
    
    dh = np.zeros((len(X),1))
    for j in range(len(X)):
        dx = np.zeros((len(X), 1))
        dx[j] = e
        y2, _ = f(X+dx, *kwargs)
        y1, _ = f(X-dx, *kwargs)
        dh[j] = (y2-y1) / (2*e)

    d = np.linalg.norm(dh-dy) / np.linalg.norm(dh+dy)
    return d
