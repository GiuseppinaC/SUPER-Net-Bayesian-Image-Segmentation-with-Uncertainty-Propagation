# updated on August 2025

# @Giuseppina Carannante
# Density Propagation for MRI Segmentation
# Implementation for BRATS DATA 

import tensorflow as tf
from tensorflow import keras
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

import os
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import math
import time, sys
import pickle
import timeit

from Brats_functions import*

plt.ioff()



#################################################################################
# The class that propagates the mean and covariance matrix of the variational distribution 
# through the First Convolution Layer     (CHECKED)
class myConv_input(tf.keras.layers.Layer):
    """y = Conv(x, w )"""
    """ x is constant and w is a r.v."""
    def __init__(self, kernel_num = 128,kernel_size=3, kernel_stride=1, padding="VALID",
                    mean_mu=0,mean_sigma=0.1, sigma_min=-12, sigma_max=-4.6):
        super(myConv_input, self).__init__()
        self.kernel_size = kernel_size #param kernel_size:  Size of the conv, kernel (Default   3)
        self.kernel_num = kernel_num #param out_channels: Number of Kernels (Required)
        self.kernel_stride = kernel_stride #param stride: Stride Length (Default   1)
        self.padding = padding 
        self.mean_mu =mean_mu
        self.mean_sigma=mean_sigma
        self.sigma_min =sigma_min
        self.sigma_max = sigma_max

    def build(self, input_shape):
        tau = 1.
        #MYinitializer = tf.random_normal_initializer(mean=self.mean_mu, stddev=self.mean_sigma, seed=None)
        MYinitializer = tf.keras.initializers.TruncatedNormal(mean=self.mean_mu, stddev=self.mean_sigma) 
        dim = self.kernel_size * self.kernel_size
        self.w_mu = self.add_weight(name='w_mu1',
            shape=(self.kernel_size,self.kernel_size, input_shape[-1], self.kernel_num),
            initializer=MYinitializer, regularizer=tf.keras.regularizers.l2(tau),
            trainable=True
        )  
        self.w_sigma = self.add_weight(name = 'w_sigma1',
            shape=(self.kernel_num,),
            initializer=tf.random_uniform_initializer(minval= self.sigma_min, maxval=self.sigma_max,  seed=None), regularizer=sigma_regularizer(dim),
            trainable=True
        )   
                  
    def call(self, inputs):
        mu_out = tf.nn.conv2d(inputs, self.w_mu, strides=self.kernel_stride, padding=self.padding )
        w_sigma = tf.math.softplus(self.w_sigma)
        vect_sigma=tf.broadcast_to(w_sigma,[self.kernel_size*self.kernel_size*self.w_mu.shape[2], self.kernel_num], name=None)
        x_patches = tf.image.extract_patches(inputs, sizes=[1, self.kernel_size, self.kernel_size, 1], strides=[1,self.kernel_stride,self.kernel_stride,1], rates=[1,1,1,1], padding = self.padding)
        # shape=[batch, new_image_size, new_image_size, kernel_size*kernel_size*num_channel]
        x_matrix = tf.reshape(x_patches,[inputs.shape[0], -1, self.kernel_size*self.kernel_size*self.w_mu.shape[2]]) #NON SICURA DELLA SHAPE w_mu[2]
        # shape=[batch, new_image_size*new_image_size, patch_size*patch_size*num_channel]
        sigma=tf.matmul(tf.math.square(x_matrix),vect_sigma)
        #shape (batch, New_image_size* New_image_size , num_kernels)
        sigma_out=tf.reshape(sigma, mu_out.shape)
        return mu_out, sigma_out
###################################################
# The class that propagates the mean and covariance matrix of the variational distribution 
# through the Intermediate Convolution Layers    
class myConv_intermediate(tf.keras.layers.Layer):
    """y = Conv(x, w )"""
    """ x and w are both r.v.
        :param kernel_num: Number of output channels       (Required)
        :param kernel_size:  Size of the conv, kernel        (Default   3)
        for 1x1 Convolution: change kernel_size = 1 and kernel_num = num_classes
        sigma_max= -2.2 (was 2.3 for MNIST)    
    """

    def __init__(self, kernel_num = 64 ,kernel_size=3, kernel_stride=1, padding="VALID",
                    mean_mu=0,mean_sigma=0.1, sigma_min=-12, sigma_max=-4.6):
        super(myConv_intermediate, self).__init__()
        self.kernel_size = kernel_size #param kernel_size:  Size of the conv, kernel (Default   3)
        self.kernel_num = kernel_num #param out_channels: Number of Kernels (Required)
        self.kernel_stride = kernel_stride #param stride: Stride Length (Default   1)
        self.padding = padding 
        self.mean_mu =mean_mu
        self.mean_sigma=mean_sigma
        self.sigma_min =sigma_min
        self.sigma_max = sigma_max

    def build(self, input_shape):
        #print(input_shape)
        tau = 1.
        #MYinitializer =tf.random_normal_initializer(mean=self.mean_mu, stddev=self.mean_sigma, seed=None)
        MYinitializer =tf.keras.initializers.TruncatedNormal(mean=self.mean_mu, stddev=self.mean_sigma)
        dim = self.kernel_size * self.kernel_size
        self.w_mu = self.add_weight(name='w_mu',
            shape=(self.kernel_size,self.kernel_size, input_shape[-1], self.kernel_num),
            initializer=MYinitializer, regularizer=tf.keras.regularizers.l2(tau),
            trainable=True
        )  
        self.w_sigma = self.add_weight(name = 'w_sigma',
            shape=(self.kernel_num,),
            initializer=tf.random_uniform_initializer(minval= self.sigma_min, maxval=self.sigma_max, seed=None), regularizer=sigma_regularizer(dim), 
            trainable=True
        )   
                  
    def call(self, inputs, sigma_input):
        mu_out = tf.nn.conv2d(inputs, self.w_mu, strides=self.kernel_stride, padding=self.padding )
        w_sigma = tf.math.softplus(self.w_sigma)
        vect_sigma=tf.broadcast_to(w_sigma,[self.kernel_size*self.kernel_size*inputs.shape[-1], self.kernel_num], name=None)
        x_patches = tf.image.extract_patches(inputs, sizes=[1, self.kernel_size, self.kernel_size, 1], strides=[1,self.kernel_stride,self.kernel_stride,1], rates=[1,1,1,1], padding = self.padding)
        # shape=[batch, new_image_size, new_image_size, kernel_size*kernel_size*num_channel]
        sigma_patches =tf.image.extract_patches(sigma_input, sizes=[1, self.kernel_size, self.kernel_size, 1], strides=[1,self.kernel_stride,self.kernel_stride,1], rates=[1,1,1,1], padding = self.padding)
        x_matrix = tf.reshape(x_patches,[inputs.shape[0], mu_out.shape[1]*mu_out.shape[1], self.kernel_size*self.kernel_size*inputs.shape[-1]]) 
        # shape=[batch, new_image_size*new_image_size, kernel_size*kernel_size*num_channel]
        sigma_matrix = tf.reshape(sigma_patches,[inputs.shape[0], mu_out.shape[1]*mu_out.shape[1], self.kernel_size*self.kernel_size*inputs.shape[-1]])
        sigma1=tf.matmul(tf.math.square(x_matrix),vect_sigma)
        #shape (batch, New_image_size* New_image_size , num_kernels)
        w_mean=tf.reshape(self.w_mu,[-1,inputs.shape[-1],self.kernel_num])
        w_mean=tf.reshape(w_mean,[-1,self.kernel_num])
        sigma2=tf.matmul(sigma_matrix, tf.math.square(w_mean))
        sigma3=tf.matmul(sigma_matrix,vect_sigma)
        sigma=sigma1+sigma2+sigma3 # shape=[batch,new_image_size*new_image_size,num_kernels]
        sigma_out=tf.reshape(sigma, mu_out.shape) # shape=[batch,new_image_size,new_image_size,num_kernels]
        
        return mu_out, sigma_out
##########################################################
#propagates mean and covariance through the upsampling layer (CHECKED)
class myupsampling(keras.layers.Layer):
    """My_Upsampling"""

    def __init__(self):
        super(myupsampling, self).__init__()
    def call(self, mu_in, sigma_in):
        mu_out = unpool(mu_in)
        sigma_out = unpool(sigma_in)
        return mu_out, sigma_out
##############################################
#propagates mean and covariance through a padding layer (we pad mean and sigma in decoder)
class mypadding(keras.layers.Layer):
    """My_Padding"""

    def __init__(self, pad_size = [2,2], sigma_fill = 0, mode= "CONSTANT"):
        super(mypadding, self).__init__()
        self.size = pad_size
        self.sigma =sigma_fill
        self.mode = mode
    def call(self, mu_in, sigma_in):
        paddings = tf.constant([[0, 0,], self.size, self.size, [0, 0,]])
        mu_out = tf.pad(mu_in,paddings, self.mode )
        sigma_out = tf.pad(sigma_in,paddings, self.mode,  constant_values=self.sigma ) #, constant_values=self.sigma
        return mu_out, sigma_out
##############################################
# Function to propagate mean and covariance through the pooling layer (CHECKED)
class mymaxpooling(keras.layers.Layer):
    """My_Max_Pooling"""

    def __init__(self):
        super(mymaxpooling, self).__init__()
    def call(self, mu_in, sigma_in):
        mu_out, argmax = tf.nn.max_pool_with_argmax(mu_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', include_batch_in_index=True) #shape=[1, new_size,new_size,num_filters[0]]  
        sigma_out= get_pooled(argmax,sigma_in)
        return mu_out, sigma_out
######################################################################
#function colled by the upsampling layer to expand the each slice of the 
#incoming tensor.
def unpool(value, name='unpool'):
    """
    :param value: A Tensor of shape [b, w,h, ch]
    :return: An upsampled Tensor of shape [b, 2*w+1, 2*h+1, ch]
    -> the output tensor will have alternating zeros and padded.
    e.g.
    input:      1 2 
                3 4

    output:  0 0 0 0 0
             0 1 0 2 0
             0 0 0 0 0
             0 3 0 4 0
             0 0 0 0 0
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out_before_pad = tf.reshape(out, out_size, name=scope)
        paddings = tf.constant([[0, 0,], [1, 0], [1, 0], [0, 0,]])
        out=tf.pad(out_before_pad, paddings, "CONSTANT")
    return out
####################################################
# Function to Pool a tensor using idices from another max pooling:   
def get_pooled(indices,other_tensor):
    #inputs: 
    #indeces: from pooling the mean tensor
    #other_tensor: (sigma) tensor to pool
    # Returns pooled sigma (other_tensor_pooled) 
    b = other_tensor.shape[0]
    w = other_tensor.shape[1]
    h=other_tensor.shape[2]
    c =other_tensor.shape[3]
    other_tensor_pooled = tf.gather(tf.reshape(other_tensor,shape= [b*w*h*c,]),indices)
    return other_tensor_pooled

#########################################################################
# This function computes the gradient of the ReLU function 
def grad_ReLU(mu_in):
  with tf.GradientTape() as g:
    g.watch(mu_in)
    out = tf.nn.relu(mu_in)
  gradi = g.gradient(out, mu_in) 
  return gradi
###########IMPLEMENTED #######################
#Propagates mean and variance through the ReLU function (CHECKED)
class myReLU(keras.layers.Layer):
    """ReLU"""

    def __init__(self):
        super(myReLU, self).__init__()
    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.relu(mu_in)
        gradi = grad_ReLU(mu_in) 
        gradi_sq=tf.math.square(gradi)
        Sigma_out = tf.math.multiply(gradi_sq,Sigma_in)
        return mu_out, Sigma_out
############################################################################
# Function to Concatenate tensors from encoder and decoder paths  (CHECKED)
class myConc(keras.layers.Layer):
    """Concatenation of downsampled (to be cropped) and upsampled feature maps
    inputs: mu_Decoder, Sigma_Decoder,mu_Encoder, Sigma_Encoder"""

    def __init__(self):
        super(myConc, self).__init__()
    def call(self, muD, SigmaD,muE, SigmaE):
        """
        Concatenation of Upsampled (from Decoder path) muD and SigmaD 
        with muE and SigmaE (corresponding Encoder tensors) respectively
        Args:
            inputs  muE and SigmaE (4-D Tensor): (N, H1, W1, C1), 
            inputs  muD and SigmaD (4-D Tensor): (N, H2, W2, C2)
        Returns:
            output (4-D Tensor): (N, H2, W2, C1 + C2)
        """
        mu_cropped = crop_tensor(muE,muD)
        sigma_cropped = crop_tensor(SigmaE,SigmaD)
        mu_out = tf.concat([muD, mu_cropped], axis=-1)
        sigma_out = tf.concat([SigmaD, sigma_cropped], axis=-1)
        return mu_out, sigma_out
#############################################################################
# The class that propagates the mean and covariance matrix of the variational distribution through the Softmax layer
class mysoftmax(keras.layers.Layer):
    """Mysoftmax pixel-wise"""

    def __init__(self):
        super(mysoftmax, self).__init__()
    def call(self, mu_in, sigma_in):
        mu_reshaped = tf.reshape(mu_in,[mu_in.shape[0], -1, mu_in.shape[3]])#shape=[Batch, size*size, num_labels]
        sigma_reshaped = tf.reshape(sigma_in,[sigma_in.shape[0], -1, sigma_in.shape[3]])#shape=[Batch, size*size, num_labels]
        mu_out = tf.nn.softmax(mu_reshaped)
        pp1 = tf.expand_dims(mu_out, axis=3)
        pp2 = tf.expand_dims(mu_out, axis=2)
        ppT = tf.matmul(pp1, pp2)
        p_diag = tf.linalg.diag(mu_out)
        grad = p_diag - ppT
        grad_sq = tf.math.square(grad)
        sigma_exp= tf.expand_dims(sigma_reshaped,axis=3)
        sigma = tf.matmul(grad_sq, sigma_exp)
        sigma_out= tf.squeeze(sigma)
        #mu_out and sigma_out shape = [bath, size*size, num_labels] for Nature (32, 4096, 3)
        return mu_out, sigma_out

########################################################################
# the log-likelihood of the objective function
# Inputs: 
#       y_pred_mean: The output Mean vector (predictive mean).
#       y_pred_sd: The output Variance vector (from predictive covariance matrix).
#       y_test: The ground truth prediction vector
# Output:
#       the expected log-likelihood term of the objective function  
def nll_gaussian(y_test, y_pred_mean, y_pred_sd): 

    eps=tf.constant(1e-3, shape=y_pred_sd.shape, name='epsilon')
    y_pred_sd_inv =  tf.math.reciprocal(tf.math.add(y_pred_sd, eps)) #Computes the reciprocal element-wise
    mu_square =  tf.math.square(y_pred_mean - y_test) 
    mu_square = tf.expand_dims(mu_square, axis=2)
    y_pred_sd_inv = tf.expand_dims(y_pred_sd_inv, axis=3) 
    loss1 =  tf.matmul(mu_square ,  y_pred_sd_inv) 
    loss = tf.reduce_mean(loss1, axis =0)
    loss = tf.reduce_mean(loss)

    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
    
    loss2 = tf.math.reduce_prod(tf.math.add(y_pred_sd, eps), axis=-1) # shape [batch,size*size]
    loss2 = tf.math.log(loss2)
    loss2 = tf.reduce_mean(loss2) 
    final_loss = 0.5*(loss+loss2)
    return final_loss
########################################IMPLEMENTED#######################
# This function computes the sigma regulirizer for the KL-divergence for convolution layers.
class sigma_regularizer(keras.regularizers.Regularizer):
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        f_s = tf.math.softplus(x) 
        return -self.strength * tf.reduce_mean(1. + tf.math.log(f_s) - f_s , axis=-1)
################################################################
# This class defines the netwrok by calling all layers          
class Density_prop_with_pad_UNET(tf.keras.Model):
  """ Required: number_of_kernels and number_of_labels"""

  def __init__(self, n_kernels, n_labels, name=None):
    super(Density_prop_with_pad_UNET, self).__init__()

    self.n_kernels = n_kernels
    self.n_out = n_labels
    self.conv_input = myConv_input(kernel_num = self.n_kernels)  # conv with 64 kernels
    self.conv1 = myConv_intermediate(kernel_num = self.n_kernels) 

    self.conv2 = myConv_intermediate(kernel_num = self.n_kernels*2)  # conv with 128 kernels
    self.conv3 = myConv_intermediate(kernel_num = self.n_kernels*2)

    self.conv4 = myConv_intermediate(kernel_num = self.n_kernels*4)  # conv with 256 kernels
    self.conv5 = myConv_intermediate(kernel_num = self.n_kernels*4)


    self.conv6 = myConv_intermediate(kernel_num = self.n_kernels*8)  # conv with 512 kernels
    self.conv7 = myConv_intermediate(kernel_num = self.n_kernels*8)

    self.conv8 = myConv_intermediate(kernel_num = self.n_kernels*16)  # conv with 1024 kernels
    self.conv9 = myConv_intermediate(kernel_num = self.n_kernels*16)

    # now starting with decoder path

    self.up1_conv2x2 = myConv_intermediate(kernel_num = self.n_kernels*8, kernel_size = 2, sigma_min=-4.6, sigma_max=-2.2)  # , sigma_min=-4.6, sigma_max=-2.2
    self.up1_conv1 = myConv_intermediate(kernel_num = self.n_kernels*8)
    self.up1_conv2 = myConv_intermediate(kernel_num = self.n_kernels*8)
    
    self.up2_conv2x2 = myConv_intermediate(kernel_num = self.n_kernels*4, kernel_size = 2, sigma_min=-4.6, sigma_max=-2.2)  # , sigma_min=-4.6, sigma_max=-2.2
    self.up2_conv1 = myConv_intermediate(kernel_num = self.n_kernels*4)
    self.up2_conv2 = myConv_intermediate(kernel_num = self.n_kernels*4)

    self.up3_conv2x2 = myConv_intermediate(kernel_num = self.n_kernels*2, kernel_size = 2)  
    self.up3_conv1 = myConv_intermediate(kernel_num = self.n_kernels*2)
    self.up3_conv2 = myConv_intermediate(kernel_num = self.n_kernels*2)


    self.up4_conv2x2 = myConv_intermediate(kernel_num = self.n_kernels, kernel_size = 2)  
    self.up4_conv1 = myConv_intermediate(kernel_num = self.n_kernels)
    self.up4_conv2 = myConv_intermediate(kernel_num = self.n_kernels)


    self.conv_final = myConv_intermediate(kernel_num = self.n_out, kernel_size = 1, sigma_min=-4.6, sigma_max=-2.2) #, sigma_min=-4.6, sigma_max=-2.2
    self.myrelu  = myReLU()
    self.myconc = myConc()
    self.mypad1 = mypadding(pad_size=[1,0], sigma_fill=0.1)
    self.mypad_up6 = mypadding(pad_size=[3,3],sigma_fill=0.1) 
    self.mypad = mypadding(sigma_fill=0.1)
    self.myups =  myupsampling()
    self.maxp =mymaxpooling()
    self.mysoftmax = mysoftmax()

  def call(self, inputs, training=True):
    # first block encoder:
    m1, s1 = self.conv_input(inputs) # 202
    m1, s1 = self.myrelu(m1, s1) # 
    m1, s1 =  self.conv1(m1, s1)  # 200
    m1, s1 = self.myrelu(m1, s1)  
    m1_p, s1_p = self.maxp(m1, s1) #  100
    # second block encoder:
    #m2, s2 = self.mypad(m1_p, s1_p) 
    m2, s2 = self.conv2(m1_p, s1_p) # 98
    m2, s2 = self.myrelu(m2, s2) 
    m2, s2 =  self.conv3(m2, s2) # 96
    m2, s2 = self.myrelu(m2, s2)  
    m2_p, s2_p = self.maxp(m2, s2) # 48
    # third block encoder:
    #m3, s3 = self.mypad(m2_p, s2_p) 
    m3, s3 = self.conv4(m2_p, s2_p) # 46
    m3, s3 = self.myrelu(m3, s3) 
    m3, s3 =  self.conv5(m3, s3) # 44
    m3, s3 = self.myrelu(m3, s3)  
    m3_p, s3_p = self.maxp(m3, s3) # 22
    # 4th block encoder:
    #m4, s4 = self.mypad1(m3_p, s3_p) 
    m4, s4 = self.conv6(m3_p, s3_p) # 20
    m4, s4 = self.myrelu(m4, s4) 
    m4, s4 =  self.conv7(m4, s4) # 18
    m4, s4 = self.myrelu(m4, s4) 
    m4_p, s4_p = self.maxp(m4, s4)  #9 
    
    # 5th block encoder:
    m5, s5 = self.mypad1(m4_p, s4_p) # 10
    m5, s5 = self.conv8(m5, s5) #8
    m5, s5 = self.myrelu(m5, s5) 
    m5, s5 =  self.conv9(m5, s5) # 6
    m5, s5 = self.myrelu(m5, s5)
    
    # first block decoder:
    d_m1, d_s1 = self.myups(m5, s5) 
    d_m1, d_s1 = self.up1_conv2x2(d_m1, d_s1) #12
    d_m1, d_s1 = self.mypad_up6(d_m1, d_s1) # 18
    d_m1, d_s1 = self.myconc(d_m1, d_s1, m4, s4)
    d_m1, d_s1 = self.up1_conv1(d_m1, d_s1 ) #16
    d_m1, d_s1 = self.myrelu(d_m1, d_s1)
    d_m1, d_s1 = self.mypad(d_m1, d_s1) #20
    d_m1, d_s1 =  self.up1_conv2(d_m1, d_s1) #18
    d_m1, d_s1 = self.myrelu(d_m1, d_s1)
    # second block decoder:
    d_m2, d_s2 = self.myups(d_m1, d_s1) 
    d_m2, d_s2 = self.up2_conv2x2(d_m2, d_s2) #36
    d_m2, d_s2 = self.mypad_up6(d_m2, d_s2) #42
    d_m2, d_s2 = self.myconc(d_m2, d_s2, m3, s3)
    d_m2, d_s2 = self.up2_conv1(d_m2, d_s2) #40
    d_m2, d_s2 = self.myrelu(d_m2, d_s2)
    d_m2, d_s2 = self.mypad(d_m2, d_s2) #44
    d_m2, d_s2 =  self.up2_conv2(d_m2, d_s2) #42
    d_m2, d_s2 = self.myrelu(d_m2, d_s2)
    # third block decoder:
    d_m3, d_s3 = self.myups(d_m2, d_s2) 
    d_m3, d_s3 = self.up3_conv2x2(d_m3, d_s3) #84
    d_m3, d_s3 = self.mypad_up6(d_m3, d_s3) #90
    d_m3, d_s3 = self.myconc(d_m3, d_s3, m2,s2)
    d_m3, d_s3 = self.up3_conv1(d_m3, d_s3) #88
    d_m3, d_s3 = self.myrelu(d_m3, d_s3)
    d_m3, d_s3 = self.mypad(d_m3, d_s3)  #92
    d_m3, d_s3 =  self.up3_conv2(d_m3, d_s3) #90
    d_m3, d_s3 = self.myrelu(d_m3, d_s3)  
    # 4th block decoder:
    d_m4, d_s4 = self.myups(d_m3, d_s3)
    d_m4, d_s4 = self.up4_conv2x2(d_m4, d_s4)  #180
    d_m4, d_s4 = self.mypad_up6(d_m4, d_s4) #186
    d_m4, d_s4 = self.myconc(d_m4, d_s4, m1,s1)
    d_m4, d_s4 = self.up4_conv1(d_m4, d_s4) #184
    d_m4, d_s4 = self.myrelu(d_m4, d_s4)
    d_m4, d_s4 = self.mypad(d_m4, d_s4) #188
    d_m4, d_s4 =  self.up4_conv2(d_m4, d_s4) #186
    d_m4, d_s4 = self.myrelu(d_m4, d_s4)
   
    m_final, s_final =  self.conv_final(d_m4, d_s4 )
    outputs, Sigma= self.mysoftmax(m_final, s_final) # output images are flattened (& sigma vector)
    #shape [batch, out_image_size*out_image_size, n_labels]
    return outputs, Sigma

    
#######################################################################################
# Main Function to train & Test with adversarial attacks - see below for noisy cases
def main_function(
    n_kernels=32,                # Number of convolutional kernels/filters (model capacity)
    output_channels=5,           # Number of output classes/channels
    batch_size=20,               # Number of samples per training batch
    epochs=100,                  # Total training epochs
    lr=0.001,                    # Initial learning rate
    lr_end=0.001,                # Final learning rate (if using a scheduler)
    kl_factor=0.00001,           # Weighting factor for KL divergence loss (regularization)
    Training=False,              # If True → start training; if False → only evaluation
    continue_training=False,     # If True → resume training from a saved checkpoint
    saved_model_epochs=100,      # Epoch number of the model checkpoint to load
    Adversarial_noise=True,      # Enable adversarial training/evaluation
    epsilon=0.0001,              # Perturbation strength for adversarial noise
    Targeted=False,              # If True → perform targeted adversarial attack
    maxAdvStep=20,               # Max steps for adversarial attack iterations
    stepSize=1,                  # Step size per adversarial iteration
    adversary_targeted_class=2,  # Target class for adversarial attack
    adv_class=3                  # Source class for attack
):
    """
    Main entry point for training, evaluation, and adversarial testing.

    Parameters:
    ----------
    n_kernels : int
        Number of convolution kernels in the network.
    output_channels : int
        Number of classes.
    batch_size : int
        Training batch size.
    epochs : int
        Total number of training epochs.
    lr : float
        Initial learning rate.
    lr_end : float
        Final learning rate for scheduling.
    kl_factor : float
        Regularization weight for KL divergence loss.
    Training : bool
        Train (True) or evaluate (False).
    continue_training : bool
        Resume training from saved checkpoint if True.
    saved_model_epochs : int
        Epoch checkpoint number to load for resuming or testing.
    Adversarial_noise : bool
        Enable adversarial training or evaluation.
    epsilon : float
        Perturbation limit for adversarial attacks.
    Targeted : bool
        If True, generate targeted adversarial examples.
    maxAdvStep : int
        Max number of steps in adversarial attack.
    stepSize : int
        Step size for adversarial attack iterations.
    adversary_targeted_class : int
        Targeted class label for adversarial attack.
    adv_class : int
        Source class for attack.
    """
    


    
    
    PATH = './Brats/saved_models_SUPER_u-Net/epoch_{}/'.format(epochs) # Modify as needed
    if not os.path.exists( PATH):
        os.makedirs(PATH)  	    
    output_size = output_channels 
    ####### Pre-processed data stored in pickles of 20 for memory constraints
    # Paths
    train_files = sorted(glob.glob("/data/giuse/Segmentation_data/Data_all/batched_data/training_batch_*.pkl"))
    val_files   = sorted(glob.glob("/data/giuse/Segmentation_data/Data_all/batched_data/validation_batch_*.pkl"))
    test_files   = sorted(glob.glob("/data/giuse/Segmentation_data/Data_all/batched_data/test_batch_*.pkl"))



    # Training dataset
    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    train_ds = train_ds.shuffle(len(train_files))       # shuffle file order
    train_ds = train_ds.interleave(
    lambda fn: tf.data.Dataset.from_tensors(fn).map(tf_load_pickle),
    cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.unbatch()                       # break apart minibatches inside each pickle
    train_ds = train_ds.shuffle(1000)                   # shuffle individual samples
    train_ds = train_ds.batch(batch_size)                       # final batch size
    train_dataset = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(val_files)
    val_ds = val_ds.interleave(
    lambda fn: tf.data.Dataset.from_tensors(fn).map(tf_load_pickle),
    cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE
    )
    val_dataset  = val_ds.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # batch_size ==20:
    n_training_steps = len(train_files)
    n_val_steps = len(val_files)
    
    n_test_steps = 100 #left for testing

    print('initializing model')    
    UNET_model = Density_prop_with_pad_UNET(n_kernels, output_size, name = 'vdp_unet') 
    num_train_steps = epochs * int(n_training_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    coef = tf.keras.metrics.MeanIoU(num_classes=output_size)

    @tf.function  # Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            logits, sigma = UNET_model(x)           
            loss_final = nll_gaussian(y, logits,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-12),
                                   clip_value_max=tf.constant(1e+3)))
            regularization_loss=tf.math.add_n(UNET_model.losses)            
            loss = loss_final + kl_factor* 0.5 *regularization_loss 
               
            gradients = tape.gradient(loss, UNET_model.trainable_weights)  
        optimizer.apply_gradients(zip(gradients, UNET_model.trainable_weights))       
        return loss, logits, sigma, gradients, regularization_loss, loss_final

    @tf.function    
    def create_adversarial_pattern(input_image, input_label):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            UNET_model.trainable = False 
            prediction, sigma = UNET_model(input_image) 
            loss_final = nll_gaussian(input_label, prediction,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+4),
                                   clip_value_max=tf.constant(1e+3)))                         
            loss = 0.5 * loss_final 
            
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
          # Get the sign of the gradients to create the perturbation
          signed_grad = tf.sign(gradient)
          return signed_grad, loss

    @tf.function    
    def create_saliency_map(input_image, target_class, tumor_structure = False):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            UNET_model.trainable = False 
            prediction, sigma = UNET_model(input_image) 
            mask_mean = get_mask(prediction, target_class, tumor =tumor_structure)
            
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(mask_mean, input_image)
          ReLU_grad = tf.nn.relu(gradient)
          return gradient, ReLU_grad, prediction 
    if Training:
        if continue_training: 
            image_size = 204
            channels =4
            UNET_model.build(input_shape=(None, image_size, image_size, channels))
            # Force build by giving input shape
            # Create a dummy input with the correct shape
            dummy_input = tf.random.normal([1, image_size, image_size, 1])

            # Forward pass: builds all variables inside the model
            _ = UNET_model(dummy_input) 
            saved_model_path = './Brats/saved_models_SUPER_u-Net/epoch_{}/'.format(saved_model_epochs)
            UNET_model.load_weights(saved_model_path + 'vdp_UNET_model.weights.h5')
        
     
        train_DICE = np.zeros(epochs) 
        valid_DICE = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
       
        start = timeit.default_timer()
        #############################
        train_dice1 =np.zeros(epochs)
        val_dice1 =np.zeros(epochs)
        train_dice2 =np.zeros(epochs)
        val_dice2 =np.zeros(epochs)
        train_dice3 =np.zeros(epochs)
        val_dice3 =np.zeros(epochs)
        
        train_hus1 =np.zeros(epochs)
        val_hus1 =np.zeros(epochs)
        train_hus2 =np.zeros(epochs)
        val_hus2 =np.zeros(epochs)
        train_hus3 =np.zeros(epochs)
        val_hus3 =np.zeros(epochs)
        ######################################
        
        for epoch in range(epochs):
            print('Epoch: ', epoch+1, '/' , epochs)           
            
          
         
            err1 = 0
            err_REG =0
            err_KL=0
            err_valid1 = 0
            
            d1=0
            val_d1 =0
            d2=0
            val_d2 =0
            d3=0
            val_d3 =0

            h1=0
            val_h1 =0
            h2=0
            val_h2 =0
            h3=0
            val_h3 =0

            tr_no_steps = 0
            va_no_steps = 0        
            train_iter = iter(train_dataset)
            val_iter   = iter(val_dataset)   
            #-------------Training--------------------
            for x, y in train_dataset:                                                       
                
                x=tf.dtypes.cast(x, tf.float32)
                y = crop_numpy_image(y,186,1)
                y =  tf.cast(y, tf.int32)  
                one_hot_y_train = tf.one_hot(y, depth=output_size)
                y_flatten = tf.reshape(one_hot_y_train,[one_hot_y_train.shape[0], -1, one_hot_y_train.shape[3]])#shape=[Batch, size*size, num_labels]               
                loss, logits, sigma, gradients, reg, kl = train_on_batch(x, y_flatten)                 
                err1+= loss
                err_REG += reg
                err_KL+=kl
                y_predictions = tf.math.argmax(logits, axis=2) # dimension of logits: [batch, size*size, num_labels]
                
                pred_reshape = tf.reshape(y_predictions, [y.shape[0], y.shape[1], y.shape[2]])
                y_arg = tf.math.argmax(y_flatten,axis=2)
                
                
                #################################################################################
                 
              
                di,di_all, hau,sens,preci ,spe= mask_tumor(y,pred_reshape)
                d1+= di
                h1+=hau
                di,di_all,hau,sens,preci ,spe= mask_core(y,pred_reshape)
                d2+= di
                h2+=hau
                di,di_all,hau,sens,preci,spe = mask_enh(y,pred_reshape)
                d3+= di
                h3+=hau

                  
                if step % 20 == 0:
                    print("\n Step:", step, "Loss:" , float(err1/(tr_no_steps + 1.)))
                    print(
                     " - New Dice Coefficients:" , float(d1/(tr_no_steps + 1.)),  float(d2/(tr_no_steps + 1.)),  float(d3/(tr_no_steps + 1.))
                     , " - Hausdorf Coefficients:" , float(h1/(tr_no_steps + 1.)),  float(h2/(tr_no_steps + 1.)),  float(h3/(tr_no_steps + 1.)))
                    
                           
                tr_no_steps+=1
            image_size = x.shape[1]         
            
            train_err[epoch] = err1/tr_no_steps
            train_dice1[epoch] = d1/tr_no_steps
            train_dice2[epoch] = d2/tr_no_steps
            train_dice3[epoch] = d3/tr_no_steps

            train_hus1[epoch] = h1/tr_no_steps
            train_hus2[epoch] = h2/tr_no_steps
            train_hus3[epoch] = h3/tr_no_steps
     
           
            
            print('Training new dice scores  ', train_dice1[epoch], train_dice2[epoch], train_dice3[epoch])
            print('Training Hausdorff scores  ', train_hus1[epoch], train_hus2[epoch], train_hus3[epoch])

            UNET_model.save_weights(PATH + 'vdp_UNET_model.weights.h5')                  
    
            #---------------Validation----------------------           
            for x, y_o in val_dataset:
                
                x = tf.dtypes.cast(x, tf.float32)
                y_crop = crop_numpy_image(y_o,186,1)
                y_crop = tf.dtypes.cast(y_crop, tf.int32)
                y =  tf.one_hot(y_crop, depth=output_size) 
                     
                logits, sigma = UNET_model(x)  
                
                y_flatten = tf.reshape(y,[y.shape[0], -1, y.shape[3]])#shape=[Batch, size*size, num_labels]            
                vloss = nll_gaussian(y_flatten, logits,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-12),
                                           clip_value_max=tf.constant(1e+3)))
                                           
                regularization_loss=tf.math.add_n(UNET_model.losses)
                total_vloss = vloss + kl_factor*0.5 *regularization_loss
                err_valid1+= total_vloss  
                y_predictions = tf.math.argmax(logits, axis=2) # dimension of logits: [batch, size*size, num_labels]
                y_arg = tf.math.argmax(y_flatten,axis=2)
                pred_reshape = tf.reshape(y_predictions, [y.shape[0], y.shape[1], y.shape[2]])
                
                
                di,di_all,hau,sens,preci,spe = mask_tumor(y_crop,pred_reshape)
                val_d1+= di
                val_h1+=hau
                di,di_all,hau,sens,preci,spe = mask_core(y_crop,pred_reshape)
                val_d2+= di
                val_h2+=hau
                di,di_all,hau,sens,preci,spe = mask_enh(y_crop,pred_reshape)
                val_d3+= di
                val_h3+=hau
                
                
     
                #################################################################################
                if step % 8 == 0:                   
                    print("Step:", step, "Loss:", float(total_vloss))
                    print(
                     " - New Dice Coefficients:" , float(val_d1/(va_no_steps+ 1.)),  float(val_d2/(va_no_steps + 1.)),  float(val_d3/(va_no_steps + 1.))
                     , " - Hausdorf Coefficients:" , float(val_h1/(va_no_steps + 1.)),  float(val_h2/(va_no_steps + 1.)),  float(val_h3/(va_no_steps + 1.)))
                    
                va_no_steps+=1
          
            
            valid_error[epoch] = err_valid1/va_no_steps
            
            
            val_dice1[epoch] = val_d1/va_no_steps
            val_dice2[epoch] = val_d2/va_no_steps
            val_dice3[epoch] = val_d3/va_no_steps

            val_hus1[epoch] = val_h1/va_no_steps
            val_hus2[epoch] = val_h2/va_no_steps
            val_hus3[epoch] = val_h3/va_no_steps

            stop = timeit.default_timer()
            print('Total Training Time: ', stop - start)
           
            
            print('------------------------------------')
            print('Training error  ', train_err[epoch])
            print('Validation error  ', valid_error[epoch]) 
            print('------------------------------------')
            
            
            print('------------------------------------')
            print('train dice scores Tumor/Core/Enhancing', train_dice1[epoch],  train_dice2[epoch],   train_dice3[epoch])
            print('Validation dice', val_dice1[epoch], val_dice2[epoch], val_dice3[epoch]) 
            print('------------------------------------')
            print('train Hausdorff scores Tumor/Core/Enhancing', train_hus1[epoch],  train_hus2[epoch],   train_hus3[epoch])
            print('Validation H. scores', val_hus1[epoch], val_hus2[epoch], val_hus3[epoch])
     
        #-----------------End Training--------------------------             
        UNET_model.save_weights(PATH + 'vdp_UNET_model.weights.h5')        
        if (epochs > 1):
            
    
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training error')
            plt.plot(valid_error,'r' , label='Validation error')
            #plt.ylim(0, 1.1)
            plt.title("Density Propagation for Segmentation with UNET")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'SUPER_UNET_Data_error.png')
            plt.close(fig)

            fig = plt.figure(figsize=(15,7))
            plt.plot(train_dice1, 'b', label='Training Dice T')
            plt.plot(val_dice1,'r' , label='Validation Dice T')
            plt.plot(train_dice2, 'royalblue', label='Training Dice C')
            plt.plot(val_dice2,'firebrick' , label='Validation Dice C')
            plt.plot(train_dice3, 'lightsteelblue', label='Training Dice E')
            plt.plot(val_dice3,'salmon' , label='Validation Dice E')
            #plt.ylim(0, 1.1)
            plt.title("Density Propagation for Segmentation with UNET")
            plt.xlabel("Epochs")
            plt.ylabel("dice coefficient")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'SUPER_UNET_Data_DICE.png')
            plt.close(fig)

            fig = plt.figure(figsize=(15,7))
            plt.plot(train_hus1, 'b', label='Training Haus T')
            plt.plot(val_hus1,'r' , label='Validation Haus T')
            plt.plot(train_hus2, 'royalblue', label='Training Haus C')
            plt.plot(val_hus2,'firebrick' , label='Validation Haus C')
            plt.plot(train_hus3, 'lightsteelblue', label='Training Haus E')
            plt.plot(val_hus3,'salmon' , label='Validation Haus E')
            #plt.ylim(0, 1.1)
            plt.title("Density Propagation for Segmentation with UNET")
            plt.xlabel("Epochs")
            plt.ylabel("Hausdorff coefficient")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'SUPER_UNET_Data_Haus.png')
            plt.close(fig)
        
                   
             
        textfile = open(PATH + 'Related_hyperparameters.txt','w')    
        textfile.write(' Image Dimension : ' +str(image_size))
        textfile.write('\n No kernels in first conv block : : ' +str(n_kernels))
        textfile.write('\n No Labels : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Learning rate : ' +str(lr)) 
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))         
        textfile.write("\n---------------------------------")          
        if Training: 
            textfile.write('\n Total run time in sec : ' +str(stop - start))
            if(epochs == 1):
                
                textfile.write("\n Averaged Training  error : "+ str( train_err))
                textfile.write("\n Averaged Validation error : "+ str(valid_error ))

                                        
            else:
                
                textfile.write("\n Averaged Training  error : "+ str(np.mean(train_err[epoch])))
                textfile.write("\n Averaged Validation error : "+ str(np.mean(valid_error[epoch])))
              
                
                textfile.write("\n Averaged Training  dice score Tumor : "+ str(np.mean(train_dice1[epoch])))
                textfile.write("\n Averaged Validation dice score Tumor: "+ str(np.mean(val_dice1[epoch]))) 
                textfile.write("\n Averaged Training  dice score Core : "+ str(np.mean(train_dice2[epoch])))
                textfile.write("\n Averaged Validation dice Core : "+ str(np.mean(val_dice2[epoch]))) 
                textfile.write("\n Averaged Training  dice score Enhancing : "+ str(np.mean(train_dice3[epoch])))
                textfile.write("\n Averaged Validation dice score Enhancing: "+ str(np.mean(val_dice3[epoch])))        
                textfile.write("\n Averaged Training  Hausdorff score Tumor: "+ str(np.mean(train_hus1[epoch])))
                textfile.write("\n Averaged Validation Hausdorff score Tumor: : "+ str(np.mean(val_hus1[epoch])))                                   
                textfile.write("\n Averaged Training  Hausdorff score Core: "+ str(np.mean(train_hus2[epoch])))
                textfile.write("\n Averaged Validation Hausdorff score Core: : "+ str(np.mean(val_hus2[epoch])))  
                textfile.write("\n Averaged Training  Hausdorff score Enh.: "+ str(np.mean(train_hus3[epoch])))
                textfile.write("\n Averaged Validation Hausdorff score Enh. : : "+ str(np.mean(val_hus3[epoch])))                                                    
        textfile.write("\n---------------------------------")                
        textfile.write("\n---------------------------------")    
        textfile.close()
    
    else:
        print('use other functions below to test with gaussian noise and noise free!')
        print('next to test with adv noise')
        # Test
        test_files = sorted(glob.glob("/data/giuse/Segmentation_data/Data_all/batched_data/test_batch_*.pkl"))
        val_ds = tf.data.Dataset.from_tensor_slices(test_files)
        val_ds = val_ds.interleave(
        lambda fn: tf.data.Dataset.from_tensors(fn).map(tf_load_pickle),
        cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE
        )
        test_dataset  = val_ds.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    

        no_test_images=2000
        t_d1=0
        t_d2=0
        t_d3=0
        t_h1=0
        t_h2=0
        t_h3=0


        image_size = 204
        out_image_size = 186
        num_channel = 4
        batch_SNR = 0
        nm_test_batches = int(no_test_images/batch_size) 
         

        if Targeted:
            test_path = 'target_{}_with_{}_adversarial_noise_{}_max_iter_{}_{}/'.format(adversary_targeted_class, adv_class, epsilon, maxAdvStep, stepSize)            
        else:
            test_path = 'test_results_non_targeted_adversarial_noise_{}/'.format(epsilon)              
        UNET_model.build(input_shape=(None, image_size, image_size, num_channel))
        # Force build by giving input shape
        # Create a dummy input with the correct shape
        dummy_input = tf.random.normal([1, image_size, image_size, 1])

        # Forward pass: builds all variables inside the model
        _ = UNET_model(dummy_input) 
        UNET_model.load_weights(PATH + 'vdp_UNET_model.weights.h5')       
        test_no_steps = 0        
        n_test_steps =100
       
        all_dice_coef1 = []
        all_dice_coef2 = []
        all_dice_coef3 = []

        true_x = np.zeros([nm_test_batches* batch_size, out_image_size, out_image_size, num_channel])
        adv_perturbations= np.zeros([nm_test_batches* batch_size, out_image_size, out_image_size, num_channel])
        true_y = np.zeros([nm_test_batches* batch_size, out_image_size, out_image_size])
        masked_y = np.zeros([nm_test_batches* batch_size, out_image_size, out_image_size])
        logits_ = np.zeros([nm_test_batches* batch_size,out_image_size, out_image_size, output_size])
    
        predicted_y = np.zeros([nm_test_batches* batch_size, out_image_size, out_image_size])
        sigma_ = np.zeros([nm_test_batches* batch_size,out_image_size, out_image_size, output_size])

      
        for x, y in test_dataset:
                
            x = tf.dtypes.cast(x, tf.float32)
             
            update_progress(step / int(no_test_images/batch_size) )
            x_crop = crop_numpy_image(x, out_image_size, 3)
            true_x[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = x_crop
            y_crop  = crop_numpy_image(y, out_image_size, 1)
            y_crop = tf.dtypes.cast(y_crop, tf.int32)
            true_y[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = y_crop
            y_arg = y_crop
            x_min = np.amin(x)
            x_max =np.amax(x)


            adv_x = x
            
            #this part commented is for vgd attack
            if Targeted:
             for advStep in range(maxAdvStep):   
            
                mask = np.ma.masked_where(y_crop==adversary_targeted_class , y_crop) # masking all enhancing to 3 (adv_class)
                masked_label=np.ma.filled(mask, fill_value=adv_class)
                masked_y[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = masked_label
                y_true_batch =tf.one_hot(masked_label, depth=output_size)
                
                y_true_batch = tf.reshape(y_true_batch,[y_true_batch.shape[0], -1, y_true_batch.shape[3]])#shape=[Batch, size*size, num_labels]      
                adv_batch, adv_loss = create_adversarial_pattern( adv_x , y_true_batch)
                adv_batch2 = crop_numpy_image(adv_batch, out_image_size, 3)
                
                adv_x = adv_x + stepSize *adv_batch
                adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
                adv_x = tf.clip_by_value(adv_x, x_min, x_max) 
             else:
                y =  tf.one_hot(y_crop, depth=output_size) 
                y_flatten = tf.reshape(y,[y.shape[0], -1, y.shape[3]])#shape=[Batch, out-size*out-size, num_labels]
                adv_batch, adv_loss = create_adversarial_pattern(adv_x, y_flatten) 
                adv_batch2 = crop_numpy_image(adv_batch, out_image_size, 3)     
                adv_x = adv_x + adv_batch
                adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
                adv_x = tf.clip_by_value(adv_x, x_min, x_max)          
            adv_perturbations[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :,:, :] = adv_batch2 
            adv_x2 = crop_numpy_image(adv_x, out_image_size, 3) 
            
            
            #####################################################################

            start = timeit.default_timer()
            logits, sigma   = UNET_model(adv_x)
            stop = timeit.default_timer()  
            logits_im_shape = tf.reshape(logits,[logits.shape[0], out_image_size,out_image_size, logits.shape[2]])
            logits_[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ),:,:,:] =logits_im_shape
            pred =tf.math.argmax(logits, axis=2)
            sigma_im_shape = tf.reshape(sigma,[logits.shape[0], out_image_size,out_image_size, logits.shape[2]])
            sigma_[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :]= sigma_im_shape 
            y_crop_h=tf.one_hot(y_crop, depth=output_size)
            y_flatten = tf.reshape(y_crop_h,[y_crop_h.shape[0], -1, y_crop_h.shape[3]])#shape=[Batch, size*size, num_labels] 
            y_arg2 =  tf.math.argmax(y_flatten,axis=2)  
            
            pred = tf.reshape(pred,[pred.shape[0], out_image_size,out_image_size])
            predicted_y[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = pred
            di,di_all,hau,sens,preci,spec  = mask_tumor(y_crop,pred)
            all_dice_coef1.append(di_all)
            t_d1+= di
            t_h1+=hau
            
        
            di,di_all,hau,sens,preci ,spec = mask_core(y_crop,pred)
            all_dice_coef2.append(di_all)
            t_d2+= di
            t_h2+=hau
           
       
            di,di_all,hau,sens,preci,spec  = mask_enh(y_crop,pred)
            all_dice_coef3.append(di_all)
            t_d3+= di
            t_h3+=hau   
           
            test_no_steps+=1 

            
            numerator = np.sum(np.square(x_crop))/batch_size*num_channel
            diff = adv_x2 - x_crop
            denominator = np.sum(np.square(diff))/batch_size*num_channel
            sig = numerator/denominator
            snr_b = np.mean(10 * np.log10(sig))
            batch_SNR+= snr_b

            
            
       
        test_d1 = t_d1/test_no_steps
        test_d2 = t_d2/test_no_steps
        test_d3 = t_d3/test_no_steps
        test_h1 = t_h1/test_no_steps
        test_h2 = t_h2/test_no_steps
        test_h3 = t_h3/test_no_steps   

        

        print('Test Dice Coefficients : ', test_d1, test_d2, test_d3) 
        print('test Hausdorff values', test_h1, test_h2, test_h3)
        ### new part to dave std in paper:
        test_std_d1 = np.nanstd(all_dice_coef1, ddof=1)
        test_std_d2 = np.nanstd(all_dice_coef2, ddof=1)
        test_std_d3 = np.nanstd(all_dice_coef3, ddof=1)

        saved_result_path =  PATH + test_path 
        if not os.path.exists(saved_result_path):
            os.makedirs(saved_result_path)
        pf = open(saved_result_path + 'uncertainty_info.pkl', 'wb')            
        pickle.dump([predicted_y,logits_, sigma_,  adv_perturbations], pf)                                                
        pf.close()
        
        
        predicted_y = tf.cast(predicted_y, tf.int32)
        variance = np.take_along_axis(sigma_, np.expand_dims(predicted_y, axis=-1), axis=-1)
        variance = tf.math.reduce_mean(variance) 
        var = variance.numpy()
        print('Output Variance',var )
        print('SNR',batch_SNR/n_test_steps)   
              
        if Targeted:
            unc, unc_b , unc_t, class1_unc, class2_unc, class3_unc, enh_unc = save_adversarial_uncertainty(saved_result_path,true_x,epsilon*adv_perturbations,logits_, true_y,sigma_,masked_y,  10, Adversarial = True )
        else:
            unc, unc_b , unc_t, class1_unc, class2_unc, class3_unc, enh_unc = save_adversarial_uncertainty(saved_result_path,true_x,epsilon*adv_perturbations,logits_, true_y,sigma_,masked_y,  10, Adversarial = True, targeted=False )
        textfile = open(saved_result_path + 'Related_hyperparameters_adversarial.txt','w')   
  
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write("\n---------------------------------") 
        textfile.write("\n Output Variance: "+ str(np.mean(var)))                     
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.write("\n Averaged Test Dice Coefficient Tumor: "+ str( test_d1))
        textfile.write("\n Averaged Test Dice Coefficient Core: "+ str( test_d2))   
        textfile.write("\n Averaged Test Dice Coefficient Enhancing: "+ str( test_d3))    
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.write("\n std Test Dice Coefficient Tumor: "+ str( test_std_d1))
        textfile.write("\n std Test Dice Coefficient Core: "+ str( test_std_d2))   
        textfile.write("\n std Test Dice Coefficient Enhancing: "+ str( test_std_d3))    
        textfile.write("\n---------------------------------")
        textfile.write("\n Averaged Test Hausdorff Coefficient Tumor: "+ str( test_h1))
        textfile.write("\n Averaged Test Hausdorff Coefficient Core: "+ str( test_h2))   
        textfile.write("\n Averaged Test Hausdorf Coefficient Enhancing: "+ str( test_h3))  
        textfile.write("\n---------------------------------")
        
        
        if Adversarial_noise:
            if Targeted:
                
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adv_class))  
                textfile.write('\n attacked class: ' + str(adversary_targeted_class))                                    
                textfile.write("\n---------------------------------")
                textfile.write('\n predictive Variance per class:') 
                textfile.write("\n"+str(unc_t))
                textfile.write("\n"+str(class1_unc)) 
                textfile.write("\n"+str(class2_unc)) 
                textfile.write("\n"+str(class3_unc)) 
                textfile.write("\n"+str(enh_unc))    
                
            else:      
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write("\n Test Time in sec (per sample) : "+ str(stop - start))  
            textfile.write('\n Adversarial Noise epsilon: '+ str(epsilon ))    
            textfile.write("\n SNR: "+ str(batch_SNR/n_test_steps))               
        textfile.write("\n---------------------------------")    
        textfile.close()
        


def testing(PATH,  labels, gaussain_noise_std, slices=4, n_kernels=32,
            Random_noise=False, Speckle=False, S_and_P=False, noise_on='B'):
    """
    Test the trained model on the given dataset, with optional noise injection.

    Parameters
    ----------
    PATH : str
        Directory path to the saved model checkpoint.
    labels : list or array
        Number of labels for evaluation.
    gaussain_noise_std : float
        Standard deviation for Gaussian noise (if applied).
    slices : int, default=4
        Number of slices/patches.
    n_kernels : int, default=32
        Number of convolution kernels (used for model reconstruction).
    Random_noise : bool, default=False
        If True, apply random noise to the input data.
    Speckle : bool, default=False
        If True, apply speckle noise to the input data.
    S_and_P : bool, default=False
        If True, apply salt-and-pepper noise to the input data.
    noise_on : str, default='T'
        Specifies the structure on which noise will be applied (options,'O' object, 'B' for Background or else to entire scan).

    

    Notes
    -----
    - The function allows controlled testing of robustness against different types of noise & levels.
    - Modify `PATH` to test different checkpoints.
    """

    print('starting testing')
    output_size = labels
    batch_size =20
    or_image_size = 204 # original image size
    image_size = 186
    num_channel = slices
    UNET_model = Density_prop_with_pad_UNET(n_kernels, output_size, name = 'vdp_unet')
    
    
    # Force build by giving input shape
    # Create a dummy input with the correct shape
    dummy_input = tf.random.normal([1, image_size, image_size, 1])

    # Forward pass: builds all variables inside the model
    _ = UNET_model(dummy_input) 
    UNET_model.build(input_shape=(None, image_size, image_size, num_channel))
    # Test                      
    test_files = sorted(glob.glob("/Segmentation_data/Data_all/batched_data/test_batch_*.pkl"))
    val_ds = tf.data.Dataset.from_tensor_slices(test_files)
    val_ds = val_ds.interleave(
    lambda fn: tf.data.Dataset.from_tensors(fn).map(tf_load_pickle),
    cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset  = val_ds.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    
     
    if Random_noise:
            test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)
    elif Speckle:
            test_path = 'test_results_speckle_noise_{}/'.format(gaussain_noise_std)
    elif S_and_P:
            test_path = 'test_results_S&P_noise_{}/'.format(gaussain_noise_std)
    else:
            test_path = 'test_results_no_noise/'
           
    UNET_model.load_weights(PATH + 'vdp_UNET_model.weights.h5')
    
    no_test_images = 2000 # saved in the test pickles (100)
   
    
    nm_test_batches = int(no_test_images/batch_size) 
    test_no_steps = 0
    
    t_d1=0
    t_d2=0
    t_d3=0
    t_h1=0
    t_h2=0
    t_h3=0


    batch_SNR = 0
    new_snr = 0
    
    signal = 0
    true_x = np.zeros([nm_test_batches* batch_size, image_size, image_size, num_channel])
    noisy_x = np.zeros([nm_test_batches* batch_size, image_size, image_size, num_channel])
    true_y = np.zeros([nm_test_batches* batch_size, image_size, image_size])
    logits_ = np.zeros([nm_test_batches* batch_size,image_size, image_size, output_size])
    
    predicted_y = np.zeros([nm_test_batches* batch_size, image_size, image_size])
    sigma_ = np.zeros([nm_test_batches* batch_size,image_size, image_size, output_size])
    gauss_std = 0
    #NEW PART TO COMPUTE STD OF DICE
    all_dice_coef1 = []
    all_dice_coef2 = []
    all_dice_coef3 = []
    for x, y in test_dataset:
                
            x = tf.dtypes.cast(x, tf.float32)

            # use next 2 lines if want image_size to be different
            x1 = crop_numpy_image(x,image_size, num_ch = 3)
            y1 = crop_numpy_image(y,image_size, num_ch = 0)
            y1 =_crop = tf.dtypes.cast(y1, tf.int32)
            ###############################
           
            true_x[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = x1 # saving cropped images
            true_y[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = y1 # saving cropped labels
            
            y_arg = y
            y =  tf.one_hot(y1, depth=output_size) 
            t_x = x1 # noise free signal 
            snr_b = 0
            snr = 0
            max_val = np.amax(x1)
            min_val = np.amin(x1)

            if Random_noise:
                noise = tf.random.normal(shape = [batch_size, or_image_size, or_image_size, num_channel], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
            elif Speckle:
                noise = tf.random.normal(shape = [batch_size, or_image_size, or_image_size, num_channel], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
                noise =  x * noise
            elif S_and_P:
                noise = salt_and_pepper(x, p = gaussain_noise_std)
            
            if noise_on == 'O':
                    y_expanded=tf.expand_dims(y_arg, axis=3)
                    y_expanded = tf.broadcast_to(y_expanded,shape = [batch_size, or_image_size, or_image_size, num_channel] )
                    mask_1 = np.ma.masked_where(y_expanded == 0, noise) # true for background
                    mask_2=np.ma.filled(mask_1, fill_value=0)  # set noise = 0 wher corresponding to background
                    x = x +  mask_2
                    #x=np.clip(x,0,1)
                    x= np.clip(x,min_val,max_val)
            elif noise_on == 'B':
                    y_expanded=tf.expand_dims(y_arg, axis=3)
                    y_expanded = tf.broadcast_to(y_expanded,shape = [batch_size, or_image_size, or_image_size, num_channel] )
                    mask_1 = np.ma.masked_where(y_expanded > 0, noise) # true for non-background (for objects)
                    mask_2=np.ma.filled(mask_1, fill_value=0)  # set noise = 0 where corresponding to object
                    x = x +  mask_2
                    #x=np.clip(x,0,1)
                    x= np.clip(x,min_val,max_val)
            else:
                    x = x +  noise # apply noise everywhere
                    #x=np.clip(x,0,1)
                    x= np.clip(x,min_val,max_val)
                
            x_crop = crop_numpy_image(x, image_size)
            numerator = np.sum(np.square(t_x))/batch_size*num_channel
            diff = x_crop - t_x
            denominator = np.sum(np.square(diff))/batch_size*num_channel
            sig = numerator/denominator
            snr = np.mean(10 * np.log10(sig))
               
                
            noisy_x[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :, :] = x_crop
            
                
            logits, sigma = UNET_model(x) 
        
            sigma_im_shape = tf.reshape(sigma,[logits.shape[0], image_size,image_size, logits.shape[2]])
            logits_im_shape = tf.reshape(logits,[logits.shape[0], image_size,image_size, logits.shape[2]])
            
            logits = tf.reshape(logits_im_shape, [y.shape[0], -1, y.shape[3]])
            #################################################

            sigma_[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :]= sigma_im_shape 
            logits_[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ),:,:,:] =logits_im_shape
            pred =tf.math.argmax(logits, axis=2)

            y_flatten = tf.reshape(y,[y.shape[0], -1, y.shape[3]])#shape=[Batch, size*size, num_labels] 
            y_arg2 =  tf.math.argmax(y_flatten,axis=2)            
            
            pred = tf.reshape(pred,[pred.shape[0], image_size,image_size])
            predicted_y[test_no_steps*batch_size:(test_no_steps+1)*(batch_size ), :, :] = pred
            di,di_all,hau, sens,preci ,spec= mask_tumor(y1,pred)
            all_dice_coef1.append(di_all)
            t_d1+= di
            t_h1+=hau
            
            di,di_all,hau, sens,preci,spec = mask_core(y1,pred)
            all_dice_coef2.append(di_all)
            t_d2+= di
            t_h2+=hau
            
            di,di_all,hau, sens,preci,spec = mask_enh(y1,pred)
            all_dice_coef3.append(di_all)
            t_d3+= di
            t_h3+=hau
           
            
            
            new_snr += snr
            
            
            if Random_noise:
                signal+=sig
                                            
            test_no_steps+=1   
    gauss_std = gauss_std/test_no_steps

    new_t_snr = new_snr/test_no_steps 

    average_signal = signal/test_no_steps
    avr_sig = average_signal
   
    test_d1 = t_d1/test_no_steps
    test_d2 = t_d2/test_no_steps
    test_d3 = t_d3/test_no_steps
    test_h1 = t_h1/test_no_steps
    test_h2 = t_h2/test_no_steps
    test_h3 = t_h3/test_no_steps
    
    ### if you want to save std:
    test_std_d1 = np.nanstd(all_dice_coef1, ddof=1)
    test_std_d2 = np.nanstd(all_dice_coef2, ddof=1)
    test_std_d3 = np.nanstd(all_dice_coef3, ddof=1)
    
    predicted_y = tf.cast(predicted_y, tf.int32)  ## make numpy array
    variance = np.take_along_axis(sigma_, np.expand_dims(predicted_y.numpy(), axis=-1), axis=-1).squeeze(axis=-1)
    variance = np.mean(variance)

    print('Test Dice Coefficients : ', test_d1, test_d2, test_d3) 
    print('test Hausdorff values', test_h1, test_h2, test_h3)
    
    if Random_noise:
            print('Test average snr signal : ',new_t_snr)
            if noise_on == 'O':
                path_full= PATH + test_path + './on_object/'
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
              
                pf = open(PATH + test_path + './on_object/uncertainty_info_on_object_noise_{}.pkl'.format(gaussain_noise_std), 'wb')
            elif noise_on == 'B':
                path_full= PATH + test_path + './on_background/'
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
                pf = open(PATH + test_path + './on_background/uncertainty_info_on_background_noise_{}.pkl'.format(gaussain_noise_std), 'wb')
            else:
                path_full= PATH + test_path + './on_all/'
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
                pf = open(PATH + test_path + './on_all/uncertainty_info_noise_{}.pkl'.format(gaussain_noise_std), 'wb')       
              
            pickle.dump([logits_, sigma_, noisy_x, true_y ], pf)                                                                           
            pf.close()
    elif Speckle:
        
            print('Test average snr signal : ',new_t_snr)
            if noise_on == 'O':
                path_full= PATH + test_path + './on_object/'
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
              
                pf = open(PATH + test_path + './on_object/uncertainty_info_on_object_noise_{}.pkl'.format(gaussain_noise_std), 'wb')
            elif noise_on == 'B':
                path_full= PATH + test_path + './on_background/'
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
                pf = open(PATH + test_path + './on_background/uncertainty_info_on_background_noise_{}.pkl'.format(gaussain_noise_std), 'wb')
            else:
                path_full= PATH + test_path + './on_all/'
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
                pf = open(PATH + test_path + './on_all/uncertainty_info_noise_{}.pkl'.format(gaussain_noise_std), 'wb')       
              
            pickle.dump([logits_, sigma_, noisy_x, true_y ], pf)                                                                           
            pf.close()
    elif S_and_P: # Salt and Pepper noise
        
            print('Test average snr signal : ',new_t_snr)
            if noise_on == 'O':
                path_full= PATH + test_path + './on_object/'
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
              
                pf = open(PATH + test_path + './on_object/uncertainty_info_on_object_noise_{}.pkl'.format(gaussain_noise_std), 'wb')
            elif noise_on == 'B':
                path_full= PATH + test_path + './on_background/'
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
                pf = open(PATH + test_path + './on_background/uncertainty_info_on_background_noise_{}.pkl'.format(gaussain_noise_std), 'wb')
            else:
                path_full= PATH + test_path + './on_all/'
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
                pf = open(PATH + test_path + './on_all/uncertainty_info_noise_{}.pkl'.format(gaussain_noise_std), 'wb')       
              
            pickle.dump([logits_, sigma_, noisy_x, true_y ], pf)                                                                           
            pf.close()
    else:
            path_full= PATH + test_path
            if not os.path.exists(path_full):
                os.makedirs(path_full)
            pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')     
           
            pickle.dump([logits_, sigma_, true_x, true_y], pf)   
            pf.close()
        
        
    if Random_noise: 
            if noise_on =="O":
                textfile = open(PATH + test_path + './on_object/Related_hyperparameters.txt','w') 
            elif noise_on == 'B': 
                textfile = open(PATH+ test_path + './on_background/Related_hyperparameters.txt','w')
            else:
                textfile = open(PATH+ test_path + './on_all/Related_hyperparameters.txt','w')  
    elif Speckle: 
            if noise_on =="O":
                textfile = open(PATH + test_path + './on_object/Related_hyperparameters.txt','w') 
            elif noise_on == 'B': 
                textfile = open(PATH+ test_path + './on_background/Related_hyperparameters.txt','w')
            else:
                textfile = open(PATH+ test_path + './on_all/Related_hyperparameters.txt','w')  
    elif S_and_P: 
            if noise_on =="O":
                textfile = open(PATH + test_path + './on_object/Related_hyperparameters.txt','w') 
            elif noise_on == 'B': 
                textfile = open(PATH+ test_path + './on_background/Related_hyperparameters.txt','w')
            else:
                textfile = open(PATH+ test_path + './on_all/Related_hyperparameters.txt','w')  
    
    else:
            textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')  
    textfile.write(' Image Dimension : ' +str(image_size))
    textfile.write('\n No Labels : ' +str(output_size))
 
                    
    textfile.write("\n---------------------------------") 
    textfile.write("\n Averaged Predictive Variance : "+ str( variance))
    textfile.write("\n---------------------------------")
    textfile.write("\n Averaged Test Dice Coefficient Tumor: "+ str( test_d1))
    textfile.write("\n Averaged Test Dice Coefficient Core: "+ str( test_d2))   
    textfile.write("\n Averaged Test Dice Coefficient Enhancing: "+ str( test_d3))    
    textfile.write("\n---------------------------------")
    textfile.write("\n---------------------------------")
    textfile.write("\n std Test Dice Coefficient Tumor: "+ str( test_std_d1))
    textfile.write("\n std Test Dice Coefficient Core: "+ str( test_std_d2))   
    textfile.write("\n std Test Dice Coefficient Enhancing: "+ str( test_std_d3))    
    textfile.write("\n---------------------------------")
    textfile.write("\n Averaged Test Hausdorff Coefficient Tumor: "+ str( test_h1))
    textfile.write("\n Averaged Test Hausdorff Coefficient Core: "+ str( test_h2))   
    textfile.write("\n Averaged Test Hausdorf Coefficient Enhancing: "+ str( test_h3))  
    textfile.write("\n---------------------------------")
    textfile.write("\n---------------------------------")
    

    if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std)) 
            textfile.write("\n---------------------------------")
            textfile.write("\n ('new') Averaged SNR : "+ str( new_t_snr))
            textfile.write("\n---------------------------------")
            textfile.write("\n Averaged Signal : "+ str( avr_sig))
            if noise_on =="O":
                textfile.write('\n Random Noise Applied on Tumor only')
            elif noise_on == 'B': 
                textfile.write('\n Random Noise Applied on the Background only')
            else:
                textfile.write('\n Random Noise on Applied everywhere')  
    if Speckle:
        
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std)) 
            textfile.write("\n---------------------------------")
            textfile.write("\n ('new') Averaged SNR : "+ str( new_t_snr))
            textfile.write("\n---------------------------------")
            textfile.write("\n Averaged Signal : "+ str( avr_sig))
            if noise_on =="O":
                textfile.write('\n Speckle Noise Applied on Tumor only')
            elif noise_on == 'B': 
                textfile.write('\n Speckle Noise Applied on the Background only')
            else:
                textfile.write('\n Speckle Noise on Applied everywhere') 
    elif S_and_P:
        
            textfile.write('\n salt and pepper ') 
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std)) 
            textfile.write("\n---------------------------------")
            textfile.write("\n ('new') Averaged SNR : "+ str( new_t_snr))
            textfile.write("\n---------------------------------")
            textfile.write("\n Averaged Signal : "+ str( avr_sig))
            if noise_on =="O":
                textfile.write('\n Speckle Noise Applied on Tumor only')
            elif noise_on == 'B': 
                textfile.write('\n Speckle Noise Applied on the Background only')
            else:
                textfile.write('\n Speckle Noise on Applied everywhere')          
    textfile.write("\n---------------------------------")    
    textfile.close()
    return test_d1, test_d2, test_d3,test_h1, test_h2, test_h3, new_t_snr, avr_sig, path_full
 
if __name__ == '__main__':
    main_function()
    # Examples of avd attacks:
    #main_function(epsilon = 0.000005, maxAdvStep=20, adversary_targeted_class=2, adv_class = 3) 
    #main_function(epsilon = 0.00001, Targeted=False)

print('##############################################')

#Example with 2 noise levels
noise = [0.005, 0.01] # enter sigma for Gaussian Noise
l = len(noise)
epochs = 100
labels = 5

PATH = './Brats/saved_models_SUPER_u-Net/epoch_{}/'.format(epochs) 
###############################################
#test with noise free test data
d1_no_noise,d2_no_noise,d3_no_noise,t1_no_noise,t2_no_noise,t3_no_noise,t_snr_no_noise,t_si_no_noise, path_no_noise =testing(PATH, labels, 0, Random_noise =False, noise_on = 'B')
mean_u_no_noise,mean_background_no_noise, mean_tumor_no_noise,class1_unc, class2_unc, class3_unc, enh_unc= save_uncertainty(path_no_noise,images_n=10, noise=0, where_noise='B')

print('done with noise free')
for i in noise:
    print('working with noise', i)
    d1_noise,d2_noise,d3_noise,t1_noise,t2_noise,t3_noise,t_snr,t_si, path =testing(PATH, labels, i,  Random_noise =True, noise_on = 'B')
    mean_u,mean_background, mean_tumor, class1_unc, class2_unc, class3_unc, enh_unc= save_uncertainty(path,images_n=10, noise=i, where_noise='B')
    
    d1_noise1,d2_noise1,d3_noise1,t1_noise1,t2_noise1,t3_noise1,t_snr1,t_si1, path=testing(PATH,  labels, i, Random_noise =True,  noise_on = 'O')
    mean_u1,mean_background1, mean_tumor1,class1_unc2, class2_unc2, class3_unc2, enh_unc2= save_uncertainty(path,images_n=10, noise=i, where_noise='O')
    
    d1_noise2,d2_noise2,d3_noise2,t1_noise2,t2_noise2,t3_noise2,t_snr2,t_si2, path=testing(PATH,  labels, i,  Random_noise =True,  noise_on = 'a')
    mean_u2,mean_background2, mean_tumor2, class1_unc3, class2_unc3, class3_unc3, enh_unc3= save_uncertainty(path,images_n=10, noise=i, where_noise='A')
    

