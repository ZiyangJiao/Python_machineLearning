
import numpy as np
from linearmodel import linearmodel
from area_under_roc_curve import area_under_roc_curve
from spamupdate import spamupdate

def spamfilter(xTe,yTe,w_trained,thresh=0.3):
        # %    def spamfilter(xTe,yTe,w_trained,thresh):
        # %    
        # % INPUT:
        # % xTe = dxn data matrix
        # % yTe = 1xn label matrix
        # % threshold = any prediction below this threshold is classified as ham
        # %
        # % OUTPUT:
        # % fpr = False positive rate
        # % tpr = True positive rate
        # % auc = Area under the ROC curve
        # %
            
        [d,n]=np.shape(xTe)


        
#         % Now go through xTe one by one
        fpr=0
        tpr=0
        allpreds=np.zeros((1,n))
        [d2,n2]=np.shape(yTe)

        for i in range(n2):
            rawpred=0  #% the raw prediction (real value)
            pred=1  #% setting prediction to 1 (either 1 or -1)
            email=xTe[:,i]
            truth=yTe[:,i]
             
#             % Do prediction here:
            rawpred=linearmodel(w_trained,email)

            if(rawpred>thresh):
                pred=1;
            else:
                pred=-1;

            
            if pred>0:
                    pstring='SPAM'
            else: 
                    pstring='GOOD'

            if truth==1:
                    tstring='SPAM'
            else: 
                    tstring='GOOD'
            
            if yTe[:,i] != pred:
                # print('Wrong: %s   TRUTH: %s \n' % (pstring,tstring));

                if pred==1:
                    fpr=fpr+1
                
                # if you made a mistake, you have the chance to update w
                w=spamupdate(w_trained,email,truth)
            else:
                # print('Correct: %s   TRUTH: %s \n' % (pstring,tstring));
                if(pred>0):
                    tpr=tpr+1
               
            allpreds[:,i] = rawpred
        
        a,b,auc=area_under_roc_curve(yTe,allpreds)
        selectture=yTe[yTe==1]
        selectfalse=yTe[yTe==-1]
        tpr=tpr*1.0/selectture.size
        fpr=fpr*1.0/selectfalse.size
        
        print ("False positive rate: %.2f%%"%(fpr*100))
        print ("True positive rate: %.2f%%"%(tpr*100))
        print ("AUC: %.2f%%"%(auc*100))
        
        return a, b, auc