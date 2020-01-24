from scipy import io
import numpy as np
from trainspamfilter import trainspamfilter
from valsplit import valsplit
from linearmodel import linearmodel

datapath='/Users/ziyangjiao/Downloads/data/'
b=np.loadtxt(datapath+'data_train/index',dtype=str)

data = io.loadmat(datapath+'data_train_default.mat')
X = data['X']
Y = data['Y']

# sanity check (processed features correspond to raw text)
for k in range(b.shape[0]):
    if b[k][0]=='ham':
        assert Y[0,k] == -1.0, "data in data_train.mat does not correspond to raw text data"
    else:
        assert Y[0,k] == 1.0, "data in data_train.mat does not correspond to raw text data"


# split the data
xTr,xTv,yTr,yTv = valsplit(X,Y)
ind = len(yTr[0])

w = trainspamfilter(xTr,yTr)

correct=[]
for i in range(len(yTv[0])):
    p=linearmodel(w,xTv[:,i])

    if p>0:
            pred='SPAM'
    else:
            pred='GOOD'

    if yTv[:,i]==1:
        truth='SPAM'
    else:
        truth='GOOD'
    if (yTv[:,i] != np.sign(p)):
        correct.append(0)
        Accuracy = sum(correct)*100.0/len(correct)
        print('\n Wrong: %s - TRUTH: %s (current accuracy: %.2f)' % (pred,truth,Accuracy))

        # show misclassified email
        file_o=open(datapath+'data_train/'+b[ind+i][1], 'rb')
        content=file_o.read()
        print('=========== misclssified email: ==============')
        print(content)
        print('===============================================')
        file_o.close()

    else:
        correct.append(1)
        #Accuracy = sum(correct)*100.0/len(correct)
        #print('Correct: %s - TRUTH: %s (current accuracy: %.2f)' % (pred,truth,Accuracy))

Accuracy = sum(correct)*100.0/len(correct)
print('\nOverall Accuracy %.2f\n'%Accuracy)
