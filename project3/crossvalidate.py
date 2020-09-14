"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
import math
from trainsvm import trainsvm

def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, 1
    errors = np.zeros((len(paras), len(Cs)))
    for i in range(0, len(paras)):
        for j in range(0, len(Cs)):
            para = paras[i]
            cv = Cs[j]
            model = trainsvm(xTr, yTr, cv, ktype, para)
            res = model(xTr)
            err = np.mean(res != yTr)
            errors[i, j] = err
            if lowest_error > err:
                lowest_error = err
                bestP = para
                bestC = cv
    
    return bestC, bestP, lowest_error, errors


    