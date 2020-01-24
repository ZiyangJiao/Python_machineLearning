import sys
from scipy.sparse import csc_matrix
#from scipy.sparse import spdiags
import scipy.io as sio
import numpy as np
import hashlib

def tokenizedata(directory='data/data_train',output='default'):

    HASHBUCKETS=2**10   # you can play with this parameter

    ind=[x.split() for x in open(directory+'/index').read().split('\n') if len(x)>0] # read in index file
    num_features = HASHBUCKETS
    num_examples = len(ind)

    Y=np.zeros((1,num_examples))
    indices = []
    indptr = [0]

    # build feature matrix
    for (num,(label,fn)) in enumerate(ind):
    	# load the data, replace returns with blanks, split into words, and hash words to integers
        text = open(directory+'/'+fn, encoding = "ISO-8859-1").read().replace('\n',' ').split()
        feature_ids=list(map(lambda e: int(hashlib.sha1(e.encode('utf-8')).hexdigest(), 16) % HASHBUCKETS, text))

        # create index and pointer lists for csc sparse matrix creation
        tmp=len(feature_ids)
        indptr.append(indptr[-1]+tmp)
        indices.extend(feature_ids)

        # create label vector
        if(label=='spam'):
            Y[0,num]=1
        else:
            Y[0,num]=-1

    data = np.ones((len(indices))) # bag-of words (bow) uses a binary representation for word presence
    X = csc_matrix((data, np.asarray(indices), np.asarray(indptr)), shape=(num_features,num_examples))

    #normalize features
    X = csc_matrix(X/X.sum(0))
    print("newly created features: "+directory+'_'+output+'.mat')
    sio.savemat(directory+'_'+output+'.mat', {'X': X,'Y': Y})

if __name__ == '__main__':
    if len(sys.argv)==1:
        tokenizedata()
    elif len(sys.argv)==2:
        tokenizedata(sys.argv[1])
    else:
        tokenizedata(sys.argv[1],sys.argv[2])
