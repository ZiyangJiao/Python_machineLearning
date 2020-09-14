"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
C   : regularization constant (in front of loss)
ktype : 'linear', 'rbf', 'polynomial'
P : parameter passed to kernel

Output:
svmclassify : a classifier, svmclassify(xTe), that returns the predictions 1 or -1 on xTe

Trains an SVM classifier with kernel (ktype) and parameters (C, P) on the data set (xTr,yTr)
"""

import numpy as np
from computeK import computeK
from generateQP import generateQP
from recoverBias import recoverBias
from cvxopt import solvers
from createsvmclassifier import createsvmclassifier

def trainsvm(xTr,yTr, C, ktype, P):    
    #print("Generate Kernel...")
    K = computeK(ktype, xTr, xTr, P)
    
    #print("Generate QP...")
    Q, p, G, h, A, b = generateQP(K, yTr, C)
    
    #print("solve QP")
    solvers.options['show_progress'] = False
    sol = solvers.qp(Q, p, G, h, A, b)
    #print('Solution status:', sol['status'])
    alphas = np.array(sol['x'])
    
    #print("Recovering bias")
    bias = recoverBias(K,yTr,alphas,C)

    #print("Creating classifier")
    svmclassify = createsvmclassifier(xTr, yTr, alphas, bias, ktype, P)
    
    return svmclassify


    
    