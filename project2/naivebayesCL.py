#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY

def naivebayesCL(x, y):
# =============================================================================
#function [w,b]=naivebayesCL(x,y);
#
#Implementation of a Naive Bayes classifier
#Input:
#x : n input vectors of d dimensions (dxn)
#y : n labels (-1 or +1)
#
#Output:
#w : weight vector
#b : bias (scalar)
# =============================================================================


    
    # Convertng input matrix x and x1 into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    
    # Pre-configuring the size of matrix X
    d,n = X.shape
    
# =============================================================================
# fill in code here
#     [CPpos1, CPneg1] = naivebayesPXY(X, y)
#     [Prpos, Prneg] = naivebayesPY(X, y)
#
#     CPpos0 = 1 - CPpos1
#     CPneg0 = 1 - CPneg1
#
#     w = np.log(np.multiply(np.divide(CPpos1, CPpos0), np.divide(CPneg0, CPneg1)))
#     b = np.asscalar(np.log(Prpos / Prneg) + np.ones([1, d]).dot(np.log(CPpos0)) - np.ones([1, d]).dot(np.log(CPneg0)))
    
    pos_y, neg_y = naivebayesPY(x, y)
    posprob, negprob = naivebayesPXY(x, y)
    b = np.log(pos_y) - np.log(neg_y)
    w = np.log(posprob) - np.log(negprob)
    
    return w,b
# =============================================================================
