from IPython import display
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
import os
import glob
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import scipy.io as sio
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor
import pickle
from scipy.stats import multivariate_normal
import sklearn.datasets, sklearn.decomposition
import cv2
from skimage.feature import greycomatrix, greycoprops

from tensorly.decomposition import parafac
from tensorflow.keras import layers
import torch
from torch.autograd import Variable

import tensorly.random
def augment(image,ndim,brightness_delta=0.01,contrast_factor=0.01,rg=0.1):
        #image = tf.image.resize(image, [ndim,ndim],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image =tf.image.adjust_brightness(image, np.random.uniform(-brightness_delta,brightness_delta))
        image =tf.image.adjust_contrast(image, np.random.uniform(1-contrast_factor,1+contrast_factor))
        image = tfa.image.rotate(image, np.random.uniform(-rg,rg), interpolation='BILINEAR')
        image = tf.image.random_crop(image, size=[ndim, ndim, 1])
        #image = tf.image.resize(image, [ndim,ndim],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image

def add_bubble(image,locx,locy,level=1,size=10):
    x=np.linspace(-2, 2, size)
    y=np.linspace(-2, 2, size)
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    imagei=image.copy()
    for i in range(size):
        for j in range(size):
            imagei[i+locx,j+locy]=image[i+locx,j+locy]+rv.pdf([x[i],y[j]])*level
    return imagei

def add_bubbles(image,ndim,level=1,size=2,bnum=10):
    imagei=image.copy()
    for bi in range(bnum):
        rsize=np.random.randint(low=size/2,high=size)
        x=np.linspace(-2, 2, rsize)
        y=np.linspace(-2, 2, rsize)
        rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
        locx=np.random.randint(low=0,high=ndim-rsize)
        locy=np.random.randint(low=0,high=ndim-rsize)
        for i in range(rsize):
            for j in range(rsize):
                imagei[i+locx,j+locy]=imagei[i+locx,j+locy]+rv.pdf([x[i],y[j]])*level
    return imagei

def add_scratches(image, num_scratches=5):
    imagei = np.copy(image)*255
    for _ in range(num_scratches):
        x1, y1 = np.random.randint(0, image.shape[1] - 1), np.random.randint(0, image.shape[0] - 1)
        x2, y2 = np.random.randint(0, image.shape[1] - 1), np.random.randint(0, image.shape[0] - 1)
        color1 = np.random.uniform(50,100)
        color=(color1,color1,color1)
        thickness = np.random.randint(1, 3)
        scratch_mask = np.zeros_like(image)
        cv2.line(scratch_mask, (x1, y1), (x2, y2), color, thickness)
        imagei = cv2.add(imagei, scratch_mask)
        imagei[np.where(imagei>255)]=255
    return imagei/255

def add_blur(image, blur_size=10):
    img = np.copy(image)*255
    #blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    blurred_img = np.random.randint(50,150,image.shape)
    mask = np.zeros_like(image)
    x1, y1 = np.random.randint(20, image.shape[1] - 20), np.random.randint(20, image.shape[0] - 20)
    mask=cv2.circle(mask, (x1, y1), blur_size, (255, 255, 255), -1)
    imagei = np.where(mask!=255, img, blurred_img)
    imagei[np.where(imagei>255)]=255
    return imagei/255

def tucker_func(npall,caseno,ndim,latent_dim = 3,real_len=37):
    np.random.seed(66)
    pred_mse=np.empty(npall.shape[0])
    recons=np.empty([npall.shape[0],ndim,ndim])
    tic=time.perf_counter()
    for i in range(real_len):
        core, factors = tucker(tl.tensor(npall[i,...]), rank=[latent_dim, latent_dim])
        recons[i,...]=tucker_to_tensor([core, factors])
        pred_mse[i]=np.square(recons[i,...]-npall[i,...]).sum()
    print('fit time')
    print(time.perf_counter()-tic)
    tic=time.perf_counter()
    for i in range(real_len+1,real_len+2):
        core, factors = tucker(tl.tensor(npall[i,...]), rank=[latent_dim, latent_dim])
        recons[i,...]=tucker_to_tensor([core, factors])
        pred_mse[i]=np.square(recons[i,...]-npall[i,...]).sum()
    print('pred time')
    print(time.perf_counter()-tic)
    vars = [pred_mse,recons]
    f=open(caseno+'.pkl','wb')
    pickle.dump(vars, f)
    f.close()
    return vars

def cp_func(npall,caseno,ndim,latent_dim = 3,real_len=37):
    np.random.seed(66)
    pred_mse=np.empty(npall.shape[0])
    recons=np.empty([npall.shape[0],ndim,ndim])
    tic=time.perf_counter()
    for i in range(real_len):
        factors = parafac(tl.tensor(npall[i,...]), rank=latent_dim)
        recons[i,...]=tl.cp_to_tensor(factors)
        pred_mse[i]=np.square(recons[i,...]-npall[i,...]).sum()
    print('fit time')
    print(time.perf_counter()-tic)
    tic=time.perf_counter()
    for i in range(real_len+1,real_len+2):
        factors = parafac(tl.tensor(npall[i,...]), rank=latent_dim)
        recons[i,...]=tl.cp_to_tensor(factors)
        pred_mse[i]=np.square(recons[i,...]-npall[i,...]).sum()
    print('pred time')
    print(time.perf_counter()-tic)
    vars = [pred_mse,recons]
    f=open(caseno+'.pkl','wb')
    pickle.dump(vars, f)
    f.close()
    return vars

def pca_func(all_flat,npall,real_len,caseno,ndim,latent_dim = 3):
    np.random.seed(66)
    tic=time.perf_counter()
    mu = np.mean(all_flat[:real_len,:], axis=0)
    pca = sklearn.decomposition.PCA()
    pca.fit(all_flat[:real_len,:])
    Xhat = np.dot(pca.transform(all_flat)[:,:latent_dim], pca.components_[:latent_dim,:])
    Xhat += mu
    recons=Xhat.reshape([Xhat.shape[0],ndim,ndim])
    print('fit time')
    print(time.perf_counter()-tic)
    tic=time.perf_counter()
    pred_z=pca.transform(all_flat)[0,:latent_dim]
    #pred_mse=np.sum((recons-npall)**2,axis=(1,2))
    print('pred time')
    print(time.perf_counter()-tic)
    vars = [recons,npall,pred_z]
    f=open(caseno+'.pkl','wb')
    pickle.dump(vars, f)
    f.close()
    return vars 

def glcm(caseno,npall,latent_dim,real_len):
    np.random.seed(66)
    tic=time.perf_counter()
    npall255=(npall*255).astype(int)
    print(npall255.max())
    dis=[2,10,30]
    ang=[0,np.pi/2,np.pi/3]
    prop=['contrast', 'dissimilarity', 'homogeneity', 'correlation', 'ASM']
    all_dim=int(len(dis)*len(ang)*5)
    all_features=np.empty([npall255.shape[0],all_dim])
    for i in range(real_len):
        gm=greycomatrix(npall255[i,...], dis, ang, levels=npall255.max()+1)
        all_features[i,]=np.vstack([
            greycoprops(gm,'energy'),
            greycoprops(gm,'dissimilarity'),
            greycoprops(gm,'homogeneity'),
            greycoprops(gm,'contrast'),
            greycoprops(gm,'correlation')
            ]).flatten()
    
    def norma(train,test):
        mu=np.mean(train,axis=0)
        sd=np.std(train,axis=0)
        for ti in range(test.shape[1]):
            test[:,ti]=(test[:,ti]-mu[ti])/sd[ti]
        return test  
    pca = sklearn.decomposition.PCA()
    norm_features=norma(all_features[:real_len,],all_features)
    pca.fit(norm_features[0:real_len,:])
    print('fit time')
    print(time.perf_counter()-tic)
    tic=time.perf_counter()
    for i in range(real_len,(real_len+1)):
        gm=greycomatrix(npall255[i,...], dis, ang, levels=npall255.max()+1)
        all_features[i,]=np.vstack([
            #greycoprops(gm,'energy'),
            #greycoprops(gm,'dissimilarity'),
            #greycoprops(gm,'homogeneity'),
            #greycoprops(gm,'ASM'),
            greycoprops(gm,'correlation')
            ]).flatten()
    print('pred time')
    print(time.perf_counter()-tic)
    features=pca.transform(norm_features)[:,:latent_dim]
    
    f=open(caseno+'.pkl','wb')
    pickle.dump([norm_features,features], f)
    f.close()
    return [norm_features,features]
    
def gan_func(caseno,filelist,ndim,train_ind,npinp_test,rand_shape=4,EPOCHS = 80,BUFFER_SIZE = 2000,BATCH_SIZE=256,num_out=3,test_len=10):
    tic=time.perf_counter()
    output_folder='D:/Users/congfang/Mercury/output/'+caseno
    isExist = os.path.exists(output_folder)
    if isExist:
        os.remove(output_folder)
        print("The old directory is deleted!")
    os.makedirs(output_folder)
    print("The new directory is created!")
    np.random.seed(66)
    real_img_size=len(filelist)
    train_size=BUFFER_SIZE
    npinp_train=np.random.uniform(0,1,[train_size,ndim,ndim,1])
    for i in range(train_size):
        k=np.random.choice(train_ind)
        fi=filelist[k]
        npinpi = imageio.imread(fi)
        npinpi=np.expand_dims(npinpi,axis=-1)
        npinpi = tf.convert_to_tensor(npinpi)
        npinp_train[i,...]=augment(npinpi,ndim=ndim)
        npinp_train[i,...]=npinp_train[i,...]/255
    train_dataset=tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(npinp_train,dtype=tf.float32)))
    train_dataset = train_dataset.batch(BATCH_SIZE)
    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(int(ndim/8)*int(ndim/8)*64, use_bias=False, input_shape=(rand_shape,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((int(ndim/8), int(ndim/8), 64)))
        assert model.output_shape == (None, int(ndim/8), int(ndim/8), 64)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        #print(model.output_shape)
        assert model.output_shape == (None, int(ndim/8), int(ndim/8), 32)

        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        #print(model.output_shape)
        assert model.output_shape == (None, int(ndim/4), int(ndim/4), 16)

        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        #print(model.output_shape)
        assert model.output_shape == (None, int(ndim/2), int(ndim/2), 8)

        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        #print(model.output_shape)
        assert model.output_shape == (None, ndim, ndim, 1)

        return model

    generator = make_generator_model()


    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(8, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[ndim, ndim, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    discriminator = make_discriminator_model()



    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return [total_loss,real_loss,fake_loss]

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


    
    noise_dim = rand_shape
    num_examples_to_generate = 16

    seed = tf.random.normal([num_examples_to_generate, noise_dim])


    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss1 = discriminator_loss(real_output, fake_output)
            disc_loss=disc_loss1[0]
            disc_loss1.append(gen_loss)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss1

    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] ,vmin=0,vmax=1)
            plt.axis('off')

        plt.savefig(output_folder+'/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()


    def train(dataset, epochs):
        diclosslist=[]
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                lo=train_step(image_batch)
            diclosslist.append(lo)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                    epoch + 1,
                                    seed)

            # Save the model every 15 epochs
            #if (epoch + 1) % 15 == 0:
            print ('Time for epoch {} is {} sec, discriminator loss is {}, real loss is {}, fake loss is {}, generator loss is {}.'.format(epoch + 1, time.time()-start,diclosslist[epoch][0],diclosslist[epoch][1],diclosslist[epoch][2],diclosslist[epoch][3]))
        checkpoint_path1 = output_folder+'/cp_generator_at_epoch_{:04d}.ckpt'.format(epoch)
        checkpoint_dir1 = os.path.dirname(checkpoint_path1)
        generator.save_weights(checkpoint_path1.format(epoch=0))
        checkpoint_path2 = output_folder+'/cp_discriminator_at_epoch_{:04d}.ckpt'.format(epoch)
        checkpoint_dir2 = os.path.dirname(checkpoint_path2)
        discriminator.save_weights(checkpoint_path2.format(epoch=0))
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                            epochs,
                            seed)
        return diclosslist

    dicloss_list = train(train_dataset, EPOCHS)
    plt.clf()
    fig =plt.gcf()
    fig.set_size_inches(4, 4)
    fig.set_dpi(100)
    losslist=np.array(dicloss_list)
    #plt.plot(losslist[:,0],label='disc loss')
    plt.plot(losslist[:,2],label='fake loss')
    plt.plot(losslist[:,1],label='real loss')
    plt.plot(losslist[:,3],label='generator loss')
    plt.legend()
    fig.savefig('Training'+caseno+'.pdf')
    
    real_len=len(train_ind)
    
    npinp_pred=np.random.uniform(0,1,[real_len,ndim,ndim,1])
    k=0
    for i in train_ind:
        fi=filelist[i]
        npinpi = imageio.imread(fi)
        npinpi=np.expand_dims(npinpi,axis=-1)
        npinpi = tf.convert_to_tensor(npinpi)
        npinpi = tf.image.random_crop(npinpi, size=[ndim, ndim, 1])
        npinp_pred[k,...]=npinpi
        npinp_pred[k,...]=npinp_pred[k,...]/255
        k=k+1
    npall=np.vstack([npinp_pred,npinp_test])
    pred2_dataset=tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(npall[0:1,...],dtype=tf.float32)))
    pred2_dataset = pred2_dataset.batch(1)
    pred_img_size=npall.shape[0]
    deci=np.random.uniform(0,0,pred_img_size)
    i=0
    print('fit time')
    print(time.perf_counter()-tic)
    tic=time.perf_counter()
    for pred_2 in pred2_dataset:
        deci[i]=discriminator(pred_2).numpy()
        i=i+1
    print('pred time')
    print(time.perf_counter()-tic)
    tn=np.array(deci[0:real_len])
    control_line=np.sort(tn)[num_out]
    plt.clf()
    fig =plt.gcf()
    fig.set_size_inches(6, 4)
    fig.set_dpi(100)
    plt.scatter(range(real_len),deci[0:real_len],color="b",label="Training images")
    plt.scatter(range(real_len,real_len+test_len),deci[(real_len):(real_len+test_len)],color="r",marker="^",label="Testing real images")
    plt.scatter(range(real_len+test_len,pred_img_size),deci[(real_len+test_len):],color="g",marker="*",label="Testing images with simulated Anomaly")
    plt.axhline(y = control_line, color = 'black', label = 'Control line')
    plt.xlabel('Image index')
    plt.ylabel('Discriminator prediction')
    plt.legend()
    fig.savefig('Discriminator_prediction'+caseno+'.pdf')


    vars = deci
    f=open('GAN decision'+caseno+'.pkl','wb')
    pickle.dump(vars, f)
    f.close()
    return deci

def cvae_func(caseno,filelist,ndim,train_ind,npinp_test,latent_dim = 16,epochs = 200,BUFFER_SIZE = 1000,batch_size=128):
  np.random.seed(66)
  tic=time.perf_counter()
  output_folder='D:/Users/congfang/Mercury/output/'+caseno
  isExist = os.path.exists(output_folder)
  if not isExist:
    os.makedirs(output_folder)
    print("The new directory is created!")

  real_img_size=len(filelist)
  train_size=BUFFER_SIZE
  npinp_train=np.random.uniform(0,1,[train_size,ndim,ndim,1])
  for i in range(train_size):
      k=np.random.choice(train_ind)
      fi=filelist[k]
      npinpi = imageio.imread(fi)
      npinpi=np.expand_dims(npinpi,axis=-1)
      npinpi = tf.convert_to_tensor(npinpi)
      npinp_train[i,...]=augment(npinpi,ndim=ndim)
      npinp_train[i,...]=npinp_train[i,...]/255
  train_dataset=tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(npinp_train,dtype=tf.float32)))
  train_dataset = train_dataset.batch(batch_size)

  
  num_conv=2
  leak_alpha=0.2

  class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
      super(CVAE, self).__init__()
      self.latent_dim = latent_dim
      self.encoder = tf.keras.Sequential(
          [
              tf.keras.layers.InputLayer(input_shape=(ndim, ndim, 1)),
              tf.keras.layers.Conv2D(
                  filters=8, kernel_size=3, strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=leak_alpha)),
              tf.keras.layers.Conv2D(
                  filters=16, kernel_size=3, strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=leak_alpha)),
              tf.keras.layers.Flatten(),
              # No activation
              tf.keras.layers.Dense(latent_dim + latent_dim),
          ]
      )

      self.decoder = tf.keras.Sequential(
          [
              tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
              tf.keras.layers.Dense(units=int(ndim/(2**num_conv))*int(ndim/(2**num_conv))*8, activation=tf.keras.layers.LeakyReLU(alpha=leak_alpha)),
              tf.keras.layers.Reshape(target_shape=(int(ndim/(2**num_conv)), int(ndim/(2**num_conv)), 8)),
              tf.keras.layers.Conv2DTranspose(
                  filters=16, kernel_size=3, strides=2, padding='same',
                  activation=tf.keras.layers.LeakyReLU(alpha=leak_alpha)),
              tf.keras.layers.Conv2DTranspose(
                  filters=8, kernel_size=3, strides=2, padding='same',
                  activation=tf.keras.layers.LeakyReLU(alpha=leak_alpha)),
              # No activation
              tf.keras.layers.Conv2DTranspose(
                  filters=1, kernel_size=3, strides=1, padding='same'),
          ]
      )

    @tf.function
    def sample(self, eps=None):
      if eps is None:
        eps = tf.random.normal(shape=(100, self.latent_dim))
      return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
      mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
      return mean, logvar

    def reparameterize(self, mean, logvar):
      eps = tf.random.normal(shape=mean.shape)
      return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=True):
      logits = self.decoder(z)
      if apply_sigmoid:
        probs = tf.sigmoid(logits)
        return probs
      return logits


  optimizer = tf.keras.optimizers.Adam(1e-3)


  def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


  def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.l2_loss(x_logit-x)
    logpx_z = -tf.nn.l2_loss(x_logit-x)
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


  @tf.function
  def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
      loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


  num_examples_to_generate = 16

  random_vector_for_generation = tf.random.normal(
      shape=[num_examples_to_generate, latent_dim])
  model = CVAE(latent_dim)

  def generate_and_save_images(model, epoch, val_sample):
    mean, logvar = model.encode(val_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i + 1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray',vmin=0,vmax=1)
      plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(output_folder+'/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

  assert batch_size >= num_examples_to_generate
  for train_batch in train_dataset.take(1):
    #generate_ind=random.sample(range(val_size),num_examples_to_generate)
    train_sample = train_batch[0:num_examples_to_generate, :, :, :]

  generate_and_save_images(model, 0, train_sample)
  elbolist=[]
  for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
      train_step(model, train_x, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for train_x in train_dataset:
      loss(compute_loss(model, train_x))
    elbo = -loss.result()
    elbolist.append(elbo)
    if np.isnan(elbo):
      break
    display.clear_output(wait=False)
    print('Epoch: {}, Validation set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, train_sample)

  checkpoint_path = output_folder+'/cp_at_epoch_{:04d}.ckpt'.format(epoch)

  checkpoint_dir = os.path.dirname(checkpoint_path)
  model.save_weights(checkpoint_path.format(epoch=0))


  def prediction_mse(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    mse = tf.nn.l2_loss(x_logit-x)
    return mse
  def latent_z(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    return z
  def latent_mean(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    return mean
  def pred_x(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    return x_logit

  pred_img_size=len(train_ind)
  npinp_pred=np.random.uniform(0,1,[pred_img_size,ndim,ndim,1])
  k=0
  for i in train_ind:
      fi=filelist[i]
      npinpi = imageio.imread(fi)
      npinpi=np.expand_dims(npinpi,axis=-1)
      npinpi = tf.convert_to_tensor(npinpi)
      npinpi = tf.image.random_crop(npinpi, size=[ndim, ndim, 1])
      npinp_pred[k,...]=npinpi
      npinp_pred[k,...]=npinp_pred[k,...]/255
      k=k+1
  npall=np.vstack([npinp_pred,npinp_test])
  pred2_dataset=tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(npall[0:1,...],dtype=tf.float32)))
  pred2_dataset = pred2_dataset.batch(1)
  pred_img_size=npall.shape[0]
  pred_mse_pred=[]
  pred_z_pred=np.random.uniform(0,0,[latent_dim,pred_img_size])
  pred_mean_pred=np.random.uniform(0,0,[latent_dim,pred_img_size])
  pred_img_list=np.random.uniform(0,0,[ndim,ndim,pred_img_size])
  i=0
  print('fit time')
  print(time.perf_counter()-tic)
  tic=time.perf_counter()
  for pred_2 in pred2_dataset:
      pred_mse_pred.append(prediction_mse(model,pred_2).numpy())
      pred_z_pred[:,i]=latent_z(model,pred_2).numpy()
      pred_mean_pred[:,i]=latent_mean(model,pred_2).numpy()
      pred_img_list[...,i]=(pred_x(model,pred_2).numpy())[0,...,0]
      i=i+1
  print('pred time')
  print(time.perf_counter()-tic)
  tic=time.perf_counter()
  vars = [pred_z_pred.transpose(),pred_img_list,pred_mse_pred,pred_mean_pred.transpose()] 
  f=open(caseno+'.pkl','wb')
  pickle.dump(vars, f)
  f.close()
  return vars 

def plot_features(caseno, pred_z, pred_mse, latent_dim):
    plt.clf()
    figure, axis = plt.subplots(latent_dim, latent_dim)
    fig =plt.gcf()
    fig.set_size_inches(latent_dim*2, latent_dim*2)
    fig.set_dpi(100)
    for i in range(latent_dim):
        for j in range(latent_dim):
            axis[i,j].scatter(pred_z[:,i],pred_z[:,j],c=pred_mse, cmap='viridis',marker='+')
    plt.show()
    plt.tight_layout()
    fig.savefig('features_'+caseno+'.pdf')

def knn_sse(caseno,pred_z,real_len,num_out=3 ,K=3,test_len=10):
    def knn(X,test_x,K):
        l2dist=(np.square(X-test_x)).mean(axis=1)
        kdist=(np.sort(l2dist))[:K]
        return np.sum(kdist)
    dist=[]
    for i in range(np.shape(pred_z)[0]):
        dist.append(knn(pred_z[:real_len,:],pred_z[i,:],K))
    f=open('knn'+caseno+'.pkl','wb')
    pickle.dump(dist, f)
    f.close()
    dn=np.array(dist[0:real_len])
    control_line=-np.sort(-dn)[num_out]
    plt.clf()
    fig =plt.gcf()
    fig.set_size_inches(6, 4)
    fig.set_dpi(100)
    plt.scatter(range(real_len),dist[0:real_len],color="b",label="Training images")
    plt.scatter(range(real_len,real_len+test_len),dist[(real_len):(real_len+test_len)],color="r",marker="^",label="Testing real images")
    plt.scatter(range(real_len+test_len,len(dist)),dist[(real_len+test_len):],color="g",marker="*",label="Testing images with simulated Anomaly")
    plt.axhline(y = control_line, color = 'black', label = 'Control line')
    plt.xlabel('Image index')
    plt.ylabel('K-nearest neighbour distance')
    plt.legend()
    fig.savefig('KNN_'+caseno+'.pdf')
    return dist

def T_val(caseno,pred_z,real_len,num_out=3,ifnorm=False,test_len=10):
    if ifnorm:
        def norma(train,test):
            mu=np.mean(train,axis=0)
            sd=np.std(train,axis=0)
            for ti in range(test.shape[1]):
                test[:,ti]=(test[:,ti]-mu[ti])/sd[ti]
            return test  
        pred_z=norma(pred_z[0:real_len,:],pred_z)
    mu_val=np.mean(pred_z[0:real_len,:],axis=0)
    tp=pred_z.transpose()
    c_val=np.cov(tp[:,0:real_len])
    T_val=np.diagonal(np.dot(np.dot(np.transpose(tp-mu_val.reshape((-1,1))),np.linalg.inv(c_val)),(tp-mu_val.reshape((-1,1)))))
    f=open('T_val'+caseno+'.pkl','wb')
    pickle.dump(T_val, f)
    f.close()
    tn=np.array(T_val[0:real_len])
    control_line=-np.sort(-tn)[num_out]
    plt.clf()
    fig =plt.gcf()
    fig.set_size_inches(6, 4)
    fig.set_dpi(100)
    plt.scatter(range(real_len),T_val[0:real_len],color="b",label="Training images")
    plt.scatter(range(real_len,real_len+test_len),T_val[(real_len):(real_len+test_len)],color="r",marker="^",label="Testing real images")
    plt.scatter(range(real_len+test_len,len(T_val)),T_val[(real_len+test_len):],color="g",marker="*",label="Testing images with simulated Anomaly")
    plt.axhline(y = control_line, color = 'black', label = 'Control line')
    plt.xlabel('Image index')
    plt.ylabel('$T^2$ statistic')
    plt.legend()
    fig.savefig('T2_'+caseno+'.pdf')
    return(T_val)

def plot_sse(pred_mse, real_len,caseno,num_out=3,test_len=10,iftwo=False):
    tn=np.array(pred_mse[0:real_len])
    control_line=-np.sort(-tn)[num_out]
    
    plt.clf()
    fig =plt.gcf()
    fig.set_size_inches(6, 4)
    fig.set_dpi(100)
    plt.scatter(range(real_len),pred_mse[0:real_len],color="b",label="Training images")
    plt.scatter(range(real_len,real_len+test_len),pred_mse[(real_len):(real_len+test_len)],color="r",marker="^",label="Testing real images")
    plt.scatter(range(real_len+test_len,len(pred_mse)),pred_mse[(real_len+test_len):],color="g",marker="*",label="Testing images with simulated Anomaly")
    plt.axhline(y = control_line, color = 'black', label = 'Control line')
    if iftwo:
        control_line1=-np.sort(-tn)[real_len-num_out]
        plt.axhline(y = control_line1, color = 'black', label = 'Control line')
    plt.xlabel('Image index')
    plt.ylabel('Reconstruction SSE')
    plt.legend()
    fig.savefig('SSE_'+caseno+'.pdf')

def plot_recons(recons, npall, real_len, caseno,test_len=10,plt_size=5):
    
    plt.clf()
    figure, axis = plt.subplots(plt_size, 10)
    
    fig =plt.gcf()
    fig.set_size_inches(plt_size*3, plt_size*2)
    fig.set_dpi(100)
    fig.suptitle('Comparison of reconstruction images')
    axis[0,1].set_title('Reconstruction')
    axis[0,3].set_title('Reconstruction')
    axis[0,0].set_title('Input')
    axis[0,2].set_title('Input')
    axis[0,4].set_title('Input')
    axis[0,5].set_title('Reconstruction')
    axis[0,8].set_title('Input')
    axis[0,9].set_title('Reconstruction')
    axis[0,6].set_title('Input')
    axis[0,7].set_title('Reconstruction')
    for i in range(plt_size):
        axis[i,1].imshow(recons[real_len+test_len-plt_size+i,...], cmap='gray',vmin=0,vmax=1)
        axis[i,0].imshow(npall[real_len+test_len-plt_size+i,...], cmap='gray',vmin=0,vmax=1)
        axis[i,3].imshow(recons[real_len+test_len+i,...], cmap='gray',vmin=0,vmax=1)
        axis[i,2].imshow(npall[real_len+test_len+i,...], cmap='gray',vmin=0,vmax=1)
        axis[i,5].imshow(recons[real_len+test_len+2*plt_size+i,...], cmap='gray',vmin=0,vmax=1)
        axis[i,4].imshow(npall[real_len+test_len+2*plt_size+i,...], cmap='gray',vmin=0,vmax=1)
        axis[i,7].imshow(recons[real_len+test_len+4*plt_size+i,...], cmap='gray',vmin=0,vmax=1)
        axis[i,6].imshow(npall[real_len+test_len+4*plt_size+i,...], cmap='gray',vmin=0,vmax=1)
        axis[i,9].imshow(recons[real_len+test_len+6*plt_size+i,...], cmap='gray',vmin=0,vmax=1)
        axis[i,8].imshow(npall[real_len+test_len+6*plt_size+i,...], cmap='gray',vmin=0,vmax=1)
        axis[i,0].axis('off') 
        axis[i,1].axis('off') 
        axis[i,2].axis('off') 
        axis[i,3].axis('off') 
        axis[i,4].axis('off') 
        axis[i,5].axis('off') 
        axis[i,6].axis('off') 
        axis[i,7].axis('off') 
        axis[i,8].axis('off') 
        axis[i,9].axis('off') 
    plt.show()
    fig.savefig('test_compare'+caseno+'.pdf')
    
def plot_test_image(npall, caseno,test_len=10,plt_size=5):
    plt.clf()
    figure, axis = plt.subplots(plt_size, 10)
    
    fig =plt.gcf()
    fig.set_size_inches(plt_size*3, plt_size*1.5)
    fig.set_dpi(100)
    fig.suptitle('Simulated images with anomalies')
    for i in range(plt_size):
        for k in range(test_len):
            axis[i,k].imshow(npall[real_len+test_len+(k-2)*plt_size+i,...], cmap='gray',vmin=0,vmax=1)
            axis[i,k].axis('off') 
    plt.show()
    fig.savefig('test_image'+caseno+'.pdf')
def pred_accuracy(pred,num_out,real_len,ifflip=False):
    if ifflip:
        pred=-pred
    tn=np.array(pred[0:real_len])
    control_line=-np.sort(-tn)[num_out]
    pred_res=pred>control_line
    return pred_res[real_len:]

def cp_opt_func(npall,real_len,latent_dim,ndim, caseno):

    tic=time.perf_counter()
    tl.set_backend('pytorch')
    random_state = 66
    rng = tl.check_random_state(random_state)
    device = 'cpu'
    tensor_list=[]
    for i in range(npall.shape[0]):
        tensori = tl.tensor(npall[i,...], device=device, requires_grad=True)
        tensor_list.append(tensori)
    weight_list = []
    for i in range(real_len):
        weighti=tl.tensor(rng.random_sample(latent_dim), device=device, requires_grad=True)
        weight_list.append(weighti)
    factors = [tl.tensor(rng.random_sample((tensori.shape[i], latent_dim)),device=device, requires_grad=True) for i in range(tl.ndim(tensori))]   

    ##opt train
    n_iter = 20000
    lr = 0.01
    penalty = 500
    eta=0.2
    pat=3
    optimizer = torch.optim.Adam(weight_list+factors, lr=lr)
    rec_error=1000000
    for j in range(1, n_iter):
        rec_error_last=rec_error
        # Important: do not forget to reset the gradients
        optimizer.zero_grad()
        rec_list=[]
        loss_list=[]
        for i in range(real_len):
            reci = tl.cp_to_tensor((weight_list[i], factors))
            rec_list.append(reci)
            lossi=tl.norm(rec_list[i] - tensor_list[i], 2)
            loss_list.append(lossi)
        loss = torch.stack(loss_list, dim=0).sum(dim=0)
        f_norm=[]
        for f in factors: 
            f_norm.append(((f.pow(2).sum(dim=0)-1).pow(2)).sum())
        loss = loss + penalty*sum(f_norm)
        loss.backward()
        optimizer.step()

        if j % 200 == 0:
            rec_error = torch.stack(loss_list, dim=0).sum(dim=0)
            if torch.abs(rec_error_last-rec_error)>eta:
                pati=0
            else:
                pati=pati+1
                if pati>pat:
                    print("Epoch {},. Rec. error: {}".format(j, rec_error))
                    print(f_norm)
                    break
        if j % 1000 == 0:    
            print("Epoch {},. Rec. error: {}".format(j, rec_error))

         
    pred_z=np.empty((npall.shape[0],latent_dim))
    recons=np.empty(npall.shape)
    pred_mse=np.empty(npall.shape[0])
    for i in range(real_len):
        pred_z[i,:]=weight_list[i].detach().numpy()
        recons[i,...] = tl.cp_to_tensor((weight_list[i], factors)).detach().numpy()
        pred_mse[i]=loss_list[i].detach().numpy()
    n_iter = 20000
    lr = 0.01
    eta=0.1
    pat=3
    print('fit time')
    print(time.perf_counter()-tic)
    tic=time.perf_counter()
    for i in range(real_len+1,real_len+2):
        print(i)
        test_weighti=tl.tensor(rng.random_sample(latent_dim), device=device, requires_grad=True)    
        optimizer = torch.optim.Adam([test_weighti], lr=lr)
        rec_error=1000000
        for j in range(1, n_iter):
            rec_error_last=rec_error
            # Important: do not forget to reset the gradients
            optimizer.zero_grad()
            reci = tl.cp_to_tensor((test_weighti, factors))
            lossi=tl.norm(reci - tensor_list[i], 2)
            loss = lossi
            loss.backward()
            optimizer.step()

            if j % 200 == 0:
                rec_error = loss
                if torch.abs(rec_error_last-rec_error)>eta:
                    pati=0
                else:
                    pati=pati+1
                    if pati>pat:
                        print("Epoch {},. Rec. error: {}".format(j, rec_error))
                        break
            if j % 1000 == 0:    
                print("Epoch {},. Rec. error: {}".format(j, rec_error))
        pred_z[i,:]=test_weighti.detach().numpy()
        pred_mse[i]=lossi.detach().numpy()
        recons[i,...] = tl.cp_to_tensor((test_weighti, factors)).detach().numpy()
    print('pred time')
    print(time.perf_counter()-tic)
    vars = [pred_mse,recons,pred_z]
    f=open(caseno+'.pkl','wb')
    pickle.dump(vars, f)
    f.close()
    return vars    

def tucker_opt_func(npall,real_len,latent_dim,ndim,caseno):
    tic=time.perf_counter()
    tl.set_backend('pytorch')
    random_state = 66
    rng = tl.check_random_state(random_state)
    device = 'cpu'
    npall_mean=np.mean(npall[:real_len,...],axis=0)
    npall_center=npall-npall_mean
    npall_mean.shape
    shape = [ndim,ndim]
    tensor_list=[]
    for i in range(npall.shape[0]):
        tensori = tl.tensor(npall_center[i,...], device=device, requires_grad=True)
        #tensori = tl.tensor(rng.random_sample(shape), device=device, requires_grad=True)
        tensor_list.append(tensori)
    ranks = [latent_dim,latent_dim]
    I0=torch.eye(ranks[0],ranks[0])
    I1=torch.eye(ranks[1],ranks[1])
    factors = [tl.tensor(rng.random_sample((tensori.shape[i], ranks[i])),device=device, requires_grad=True) for i in range(tl.ndim(tensori))]  
    ##
    n_iter = 20000
    lr = 0.005
    penalty = 1e7
    eta=0.0001
    pat=2
    optimizer = torch.optim.Adam(factors, lr=lr)
    utu_error=10000
    for j in range(1, n_iter):
        utu_error_last=utu_error
        # Important: do not forget to reset the gradients
        optimizer.zero_grad()
        norm_list=[]
        core_list = []
        for i in range(real_len):
            corei=tl.tenalg.mode_dot(tl.tenalg.mode_dot(tensor_list[i],factors[0],mode=0,transpose=True),factors[1],mode=1,transpose=True)
            core_list.append(corei)
            norm_list.append(corei.pow(2).sum())
        utu0=torch.matmul(torch.transpose(factors[0],0,1),factors[0])
        utu1=torch.matmul(torch.transpose(factors[1],0,1),factors[1])
        utu_error=(utu0-I0).pow(2).sum()+(utu1-I1).pow(2).sum()
        norm_z= torch.stack(norm_list, dim=0).sum()
        loss =- norm_z+penalty*utu_error
        loss.backward()
        optimizer.step()
        if j % 200 == 0:    
            print("Epoch {},. norm: {}, utu: {}".format(j, norm_z, utu_error))
        if j % 100 == 0:
            if torch.abs(utu_error_last-utu_error)>eta:
                pati=0
            else:
                pati=pati+1
                if pati>pat:
                    break    
    print("Epoch {},. norm: {}, utu: {}".format(j, norm_z, utu_error))
            
    pred_z=np.empty((npall.shape[0],ranks[0]*ranks[1]))
    print('fit time')
    print(time.perf_counter()-tic)
    tic=time.perf_counter()
    for i in range(1):
        corei=tl.tenalg.mode_dot(tl.tenalg.mode_dot(tensor_list[i],factors[0],mode=0,transpose=True),factors[1],mode=1,transpose=True)
        pred_z[i,:]=corei.detach().numpy().flatten()
    print('pred time')
    print(time.perf_counter()-tic)
    f=open(caseno+'.pkl','wb')
    pickle.dump(pred_z, f)
    f.close() 
    return pred_z   

def plot_norm(pred_mse, real_len,caseno,lab,ifflip=False):
    if ifflip:
        pred_mse=-pred_mse
    tn=np.array(pred_mse[0:real_len])
    control_mean=np.mean(tn)
    control_std=np.std(tn)
    if ifflip:
        control_std=-control_std
    control_line=control_mean+control_std*3
    pred_res=pred_mse>control_line
    plt.clf()
    fig =plt.gcf()
    fig.set_size_inches(6, 4)
    fig.set_dpi(100)
    plt.scatter(range(real_len),pred_mse[0:real_len],color="b",label="Training images")
    plt.scatter(range(real_len,real_len+5),pred_mse[(real_len):(real_len+5)],color="r",marker="^",label="Testing real images")
    plt.scatter(range(real_len+5,len(pred_mse)),pred_mse[(real_len+5):],color="g",marker="*",label="Testing images with simulated Anomaly")
    plt.axhline(y = control_line, color = 'black', label = 'Control line')
    plt.xlabel('Image index')
    plt.ylabel(lab)
    plt.legend()
    fig.savefig('norm_'+caseno+'.pdf')
    return pred_res[real_len:]
# augment(image,ndim,brightness_delta=0.01,contrast_factor=0.01,rg=0.1)
# tucker_func(npall,caseno,ndim,latent_dim = 3): [recons, pred_mse]
# cp_func(npall,caseno,ndim,latent_dim = 3):[recons, pred_mse]
# pca_func(all_flat,npall,real_len,caseno,ndim,latent_dim = 3) [recons, pred_mse,pred_z]
# glcm(caseno,npall,latent_dim,real_len) [norm_features,features]
# gan_func(caseno,filelist,ndim,train_ind,npinp_test,rand_shape=4,EPOCHS = 80,BUFFER_SIZE = 2000,BATCH_SIZE=256,num_out=3)[deci]
# cvae_func(caseno,filelist,ndim,train_ind,npinp_test,latent_dim = 16,epochs = 200,BUFFER_SIZE = 1000,batch_size=128) [pred_z_val.transpose(),pred_img_list,pred_mse_val,pred_mean_val.transpose()] 
# plot_features(caseno, pred_z, pred_mse, latent_dim)
# knn_sse(caseno,pred_z,real_len,num_out=3 ,K=3)
# T_val(caseno,pred_z,real_len,num_out=3,ifnorm=False):
# plot_sse(pred_mse, real_len,caseno,num_out=3):
# plot_recons(recons, npall, real_len, caseno):
