
import numpy as np
from ridge import ridge
from hinge import hinge
from logistic import logistic
from grdscent import grdescent
from scipy import io

def trainspamfilter(xTr,yTr):

    #
    # INPUT:
    # xTr
    # yTr
    #
    # OUTPUT: w_trained
    #
    # Consider optimizing the input parameters for your loss and GD!

    f = lambda w : ridge(w,xTr,yTr,0.13)
    # f = lambda w: logistic(w, xTr, yTr)
    w_trained = grdescent(f,np.zeros((xTr.shape[0],1)),1e-04,1000)
    io.savemat('w_trained.mat', mdict={'w': w_trained})
    return w_trained
