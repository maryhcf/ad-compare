# %%
from IPython import display
import torch
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pickle
import PIL
import time
import os
import glob
import scipy.io as sio
from scipy.stats import multivariate_normal
import sklearn.datasets, sklearn.decomposition
import cv2
from skimage.feature import greycomatrix, greycoprops
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor
from tensorly.decomposition import parafac
from torch.autograd import Variable
import tensorly.random
import functions_file as ff

PATH=**DATAPATH**


dim=100
ndim=96
latent_dim=4
num_out=2
plt.gray()
method_list=['Tucker_SSE', 'Tucker_opt_$T^2$','Tucker_opt_KNN','CP_SSE','CP_opt_SSE','CP_opt_$T^2$','CP_opt_KNN',
    'PCA_SSE','PCA_$T^2$', 'PCA_KNN', 'GLCM_$T^2$','GLCM_KNN','GLCM_PCA_$T^2$','GLCM_PCA_KNN','GAN','CVAE_SSE','CVAE_$T^2$','CVAE_KNN']
#######simulate anomalies###########
filelist= glob.glob(PATH+'*'+".jpg")
caseno=**DATAINDEX**
np.random.seed(66)
train_ind,test_ind = train_test_split(range(len(filelist)), test_size = 10,random_state=6)
np.random.seed(66)
real_img_size=len(filelist)
npinp_train=np.random.uniform(0,1,[len(train_ind),ndim,ndim,1])
for i in range(len(train_ind)):
    fi=filelist[train_ind[i]]
    npinpi = imageio.imread(fi)
    npinpi=np.expand_dims(npinpi,axis=-1)
    npinpi = tf.convert_to_tensor(npinpi)
    npinpi = tf.image.random_crop(npinpi, size=[ndim, ndim, 1])
    npinp_train[i,...]=npinpi/255
train_flat=npinp_train.reshape(npinp_train.shape[0], -1)
test_size=50
npinp_test=np.random.uniform(0,1,[test_size,ndim,ndim,1])
np.random.seed(66)
for i in range(test_size):
    if i<10:
        fi=filelist[test_ind[i]]
        npinpi = imageio.imread(fi)
        npinpi=np.expand_dims(npinpi,axis=-1)
        npinpi = tf.convert_to_tensor(npinpi)
        npinpi = tf.image.random_crop(npinpi, size=[ndim, ndim, 1])
        npinp_test[i,...]=npinpi.numpy()/255
        org=npinpi.numpy()/255
    elif i<20:
        ii=i+5
        isize=(ii%5+5)*5
        ilevel=(ii//5+4)*0.2
        npinp_test[i,...,0]=ff.add_bubble(org[...,0],size=isize,level=ilevel,locx=40,locy=40)  
    elif i<30:
        ii=i+5
        ibnum=ii%5+5
        ilevel=(ii//5+2)*0.1
        npinp_test[i,...,0]=ff.add_bubbles(org[...,0],ndim,size=30,level=ilevel,bnum=ibnum)
    elif i<40:
        ilinenum=i%5+1
        npinp_test[i,...,0]=ff.add_scratches(org[...,0],num_scratches=ilinenum)
    else:
        iblursize=(i%5)*2+4
        npinp_test[i,...,0]=ff.add_blur(org[...,0],blur_size=iblursize)
test_flat=npinp_test.reshape(npinp_test.shape[0], -1)
npall=np.vstack([npinp_train,npinp_test])[...,0]
all_flat=np.vstack([train_flat,test_flat])
all_img_size=len(train_ind)+test_size
real_len=len(train_ind)
train_flat=npinp_train.reshape(npinp_train.shape[0], -1)
#######fit/training###########
##tucker
tucker_res=ff.tucker_func(npall,'tucker'+caseno,ndim,latent_dim=latent_dim)
#tucker_opt
ff.tucker_opt_func(npall,real_len,latent_dim,ndim,'tucker_opt_'+caseno)
##cp
cp_res=ff.cp_func(npall,'cp'+caseno,ndim,latent_dim=latent_dim)
##cp_opt
cp_opt_res=ff.cp_opt_func(npall,real_len,latent_dim,ndim, 'cp_opt_'+caseno)
##pca
pca_res=ff.pca_func(all_flat,npall,real_len,'pca'+caseno,ndim,latent_dim=latent_dim)
##glcm
glcm_res=ff.glcm('glcm'+caseno,npall,latent_dim=latent_dim,real_len=real_len)
##gan
gan_res=ff.gan_func('gan_sim'+caseno,filelist,ndim,train_ind,npinp_test,rand_shape=latent_dim,EPOCHS = 40,BUFFER_SIZE = 2000,BATCH_SIZE=8,num_out=6)
##cvae
cvae_res=ff.cvae_func('cvae'+caseno,filelist,ndim,train_ind,npinp_test,latent_dim = latent_dim,epochs = 200,BUFFER_SIZE = 1000,batch_size=128)

display.clear_output(wait=True)

#####Prediction############
label=np.empty([len(method_list),50])
Prediction=np.empty([len(method_list),50])
### tucker_SSE
Prediction[0,:]=tucker_res[0][real_len:]
label[0,:]=ff.pred_accuracy(tucker_res[0],num_out,real_len,ifflip=False)
### Tucker_opt
# Tucker_opt_$T^2$
tucker_opt_tval=ff.T_val('tucker_opt'+caseno,tucker_opt_res,real_len,num_out=num_out,ifnorm=False)
Prediction[1,:]=tucker_opt_tval[real_len:]
label[1,:]=ff.pred_accuracy(tucker_opt_tval,num_out,real_len)
# Tucker_opt_KNN
tucker_opt_knn=ff.knn_sse('tucker_opt'+caseno,tucker_opt_res,real_len,num_out=num_out,K=2)
Prediction[2,:]=tucker_opt_knn[real_len:]
label[2,:]=ff.pred_accuracy(tucker_opt_knn,num_out,real_len)
### CP_SSE
Prediction[3,:]=cp_res[0][real_len:]
label[3,:]=ff.pred_accuracy(cp_res[0],num_out,real_len,ifflip=False)
### cp_opt
# cp_opt_sse
Prediction[4,:]=cp_opt_res[0][real_len:]
label[4,:]=ff.pred_accuracy(cp_opt_res[0],num_out,real_len,ifflip=False)
# cp_opt_$T^2$
cp_opt_tval=ff.T_val('cp_opt'+caseno,cp_opt_res[2],real_len,num_out=num_out,ifnorm=False)
Prediction[5,:]=cp_opt_tval[real_len:]
label[5,:]=ff.pred_accuracy(cp_opt_tval,num_out,real_len)
# cp_opt_KNN      
cp_opt_knn=ff.knn_sse('cp_opt'+caseno,cp_opt_res[2],real_len,num_out=num_out,K=1)
Prediction[6,:]=cp_opt_knn[real_len:]
label[6,:]=ff.pred_accuracy(cp_opt_knn,num_out,real_len)
### PCA
### PCA_SSE
Prediction[7,:]=pca_res[0][real_len:]
label[7,:]=ff.pred_accuracy(pca_res[0],num_out,real_len,ifflip=False)
# PCA_$T^2$
pca_tval=ff.T_val('pca'+caseno,pca_res[3],real_len,num_out=num_out,ifnorm=False)
Prediction[8,:]=pca_tval[real_len:]
label[8,:]=ff.pred_accuracy(pca_tval,num_out,real_len)
# PCA_KNN
pca_knn=ff.knn_sse('pca'+caseno,pca_res[3],real_len,num_out=num_out,K=5)
Prediction[9,:]=pca_knn[real_len:]
label[9,:]=ff.pred_accuracy(pca_knn,num_out,real_len)
###glcm
# GLCM_$T^2$
glcm_tval=ff.T_val('glcm'+caseno,glcm_res[0],real_len,num_out=num_out,ifnorm=False)
Prediction[10,:]=glcm_tval[real_len:]
label[10,:]=ff.pred_accuracy(glcm_tval,num_out,real_len)
# GLCM_KNN
glcm_knn=ff.knn_sse('glcm'+caseno,glcm_res[0],real_len,num_out=num_out,K=3)
Prediction[11,:]=glcm_knn[real_len:]
label[11,:]=ff.pred_accuracy(glcm_knn,num_out,real_len)
# GLCM_pca_$T^2$
glcm_pca_tval=ff.T_val('glcm_pca'+caseno,glcm_res[1],real_len,num_out=num_out,ifnorm=False)
Prediction[12,:]=glcm_pca_tval[real_len:]
label[12,:]=ff.pred_accuracy(glcm_pca_tval,num_out,real_len)
# GLCM_pca_KNN
glcm_pca_knn=ff.knn_sse('glcm_pca'+caseno,glcm_res[1],real_len,num_out=num_out,K=3)
Prediction[13,:]=glcm_pca_knn[real_len:]
label[13,:]=ff.pred_accuracy(glcm_pca_knn,num_out,real_len)
###GAN
Prediction[14,:]=-gan_res[real_len:]
label[14,:]=ff.pred_accuracy(gan_res,num_out,real_len,ifflip=True)
### CVAE
### CVAE_SSE
Prediction[15,:]=cvae_res[2][real_len:]
label[15,:]=ff.pred_accuracy(cvae_res[2],num_out,real_len)
### CVAE_$T^2$
cvae_tval=ff.T_val('cvae'+caseno1,cvae_res[0],real_len,num_out=num_out,ifnorm=False)
Prediction[16,:]=cvae_tval[real_len:]
label[16,:]=ff.pred_accuracy(cvae_tval,num_out,real_len)
### CVAE_KNN
cvae_knn=ff.knn_sse('cvae'+caseno1,cvae_res[0],real_len,num_out=num_out,K=3)
Prediction[17,:]=cvae_knn[real_len:]
label[17,:]=ff.pred_accuracy(cvae_knn,num_out,real_len)
f=open('sim_prediction.pkl','wb')
pickle.dump([Prediction,label],f)
f.close()




