#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY

def naivebayes(x, y, x1):
# =============================================================================
#function logratio = naivebayes(x,y,x1);
#
#Computation of log P(Y|X=x1) using Bayes Rule
#Input:
#x : n input vectors of d dimensions (dxn)
#y : n labels (-1 or +1)
#x1: input vector of d dimensions (dx1)
#
#Output:
#logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))
# =============================================================================


    
    # Convertng input matrix x and x1 into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    X1= np.matrix(x1)
    
    # Pre-configuring the size of matrix X
    d,n = X.shape
    
# =============================================================================
# fill in code here
#     pos_y, neg_y = naivebayesPY(x, y)
#     posprob, negprob = naivebayesPXY(x, y)
#     logpos = np.log(pos_y) + np.sum((x1 * np.log(posprob)) + np.sum((1-x1)*np.log(1-posprob)))
#     logneg = np.log(neg_y) + np.sum((x1 * np.log(negprob)) + np.sum((1-x1)*np.log(1-negprob)))
#     logratio = logpos - logneg
    [CPpos1, CPneg1] = naivebayesPXY(X, y)
    [Prpos, Prneg] = naivebayesPY(X, y)

    CPpos0 = 1 - CPpos1
    CPneg0 = 1 - CPneg1

    a = X1.T.dot(np.log(np.divide(CPpos1, CPneg1)))
    b = (1 - X1.T).dot(np.log(np.divide(CPpos0, CPneg0)))

    logCPRatioSum = a + b
    logratio = np.asscalar(logCPRatioSum + np.log(np.divide(Prpos, Prneg)))
    return logratio
# =============================================================================
