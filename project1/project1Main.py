from spamfilter import spamfilter
from trainspamfilter import trainspamfilter
from valsplit import valsplit
from checkgradHingeAndRidge import checkgradHingeAndRidge
from scipy import io
import numpy as np

# load the data:
# data = io.loadmat('data/data_train_default.mat')
data = io.loadmat('/Users/ziyangjiao/Downloads/data/data_train_default.mat')
X = data['X']
Y = data['Y']

# split the data:
xTr,xTv,yTr,yTv = valsplit(X,Y)

# train spam filter with settings and parameters in trainspamfilter.py
w_trained = trainspamfilter(xTr,yTr)

# evaluate spam filter on test set using default threshold
spamfilter(xTv,yTv,w_trained)
