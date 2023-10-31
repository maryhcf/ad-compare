
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
from tabulate import tabulate


f=open('sim_prediction.pkl','rb')
res=pickle.load(f)
f.close()
Prediction=res[0]
label=res[1]

y_true=[0]*10+[1]*40
all_res=np.empty([18,7])
for mi in range(18):
    mean_acc=(1-label[mi,0:10].mean()+4*label[mi,10:].mean())/5
    all_res[mi,:]=[1-label[mi,0:10].mean(),label[mi,10:20].mean(),label[mi,20:30].mean(),label[mi,30:40].mean(),label[mi,40:50].mean(),mean_acc,roc_auc_score(y_true, Prediction[mi,:])]
print(tabulate(all_res))


