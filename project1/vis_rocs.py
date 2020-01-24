from valsplit import valsplit
from scipy import io
from ridge import ridge
from hinge import hinge
from logistic import logistic
from grdscent import grdescent
from linearmodel import linearmodel
from area_under_roc_curve import area_under_roc_curve
import numpy as np
import matplotlib.pyplot as plt


def vis_rocs():
        data = io.loadmat('data/data_train_default.mat')
        X = data['X']
        Y = data['Y']
        xTr,xTv,yTr,yTv = valsplit(X,Y)

        MAXITER=100;
        STEPSIZE=1e-01;




#         %% Ridge Regression
        d,n = xTr.shape
        f = lambda w : ridge(w,xTr,yTr,0.1)
        ws=grdescent(f,np.zeros((d,1)),STEPSIZE,MAXITER)

        preds=linearmodel(ws,xTv)
        fpr,tpr,sqauc = area_under_roc_curve(yTv,preds)

        plt.plot(fpr,tpr,color="blue", linewidth=2.0,label="ridge")



#         %% Hinge Loss
        d,n = xTr.shape
        f = lambda w : hinge(w,xTr,yTr,0.1)
        wh=grdescent(f,np.zeros((d,1)),STEPSIZE,MAXITER)
        preds=linearmodel(wh,xTv);
        fpr,tpr,hinauc=area_under_roc_curve(yTv,preds)

        plt.plot(fpr,tpr,color="green", linewidth=2.0,label="hinge")


#         %% Logistic Regression
        d,n = xTr.shape
        f = lambda w : logistic(w,xTr,yTr)
        wl=grdescent(f,np.zeros((d,1)),STEPSIZE,MAXITER)
        preds=linearmodel(wl,xTv);
        fpr,tpr,logauc=area_under_roc_curve(yTv,preds)

        plt.plot(fpr,tpr,color="red", linewidth=2.0,label="logistic")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='upper right');



        print("Hinge loss: Area under the curve: %.2f"%hinauc)
        print("Logistic loss: Area under the curve: %.2f"%logauc)
        print("Squared loss: Area under the curve: %.2f"%sqauc)
        plt.show()
        return
vis_rocs()
