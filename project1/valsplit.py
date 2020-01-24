import math
import numpy as np
def valsplit(X,Y):
    [d,n] = np.shape(X)
    part = math.ceil(n*0.8)
    part = int(part)
    xTr = X[:,0:part]
    xTr = xTr.toarray()
    xTv = X[:,part:n]
    xTv = xTv.toarray()
    yTr = Y[:,0:part]
    yTv = Y[:,part:n]
    return xTr,xTv,yTr,yTv
