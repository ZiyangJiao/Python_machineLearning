
import numpy as np

def area_under_roc_curve(labels, scores):
    labels = labels.T
    scores = scores.T
    if labels.shape[0] != scores.shape[0]:
        print("error: Length of labels must be equal in order to compute area under ROC curve")
    

    indx = np.argsort(-scores,axis=0)
    
    roc_y = labels[indx,:]

    X = np.cumsum(roc_y == -1)*1.0/sum(roc_y == -1)
    Y = np.cumsum(roc_y == 1)*1.0/sum(roc_y == 1)
    leng = roc_y.shape[0]
    X = X.T
    Y = Y.T
    AUC = sum( (X[ 1:leng-1,: ] - X[ 0:(leng-2),:] )*1.0* Y[1:leng-1,:] )
    return X,Y,AUC
