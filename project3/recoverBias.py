"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    bias = 0
    pos = 0
    dis = 0.5*C
    for i in range(0, len(alphas)):
        alpha = alphas[i]
        tmp = np.abs(alpha - 0.5*C)
        if dis > tmp:
            dis = tmp
            pos = i
    bias = yTr[pos] - np.transpose(alphas)*np.transpose(yTr) @ K[:, pos]
    
    return bias