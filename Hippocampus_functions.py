
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import math
import time, sys
import pickle
import timeit
plt.ioff()
from mpl_toolkits import axes_grid1

def softplus(x):
    return np.log(1 + np.exp(x))   
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def save_adversarial_uncertainty(path,truex,adv, logits, truey,sigma,masked, images_n, Adversarial = True, targeted=True):
    print('going over function uncertainty for adversarial')
     
    path_full= path + './test_images/'
    if not os.path.exists(path_full):
        os.makedirs(path_full)

    #sigma = softplus(sigma)
    #N = truex.shape[0] 
    #N = 1147 #original images not augmented
    N = 403
    predict = np.argmax(logits, axis = -1)
    predict2 = predict
    mask_uncert = np.ma.masked_invalid(sigma)
    uncert= np.take_along_axis(sigma, np.expand_dims(predict, axis=-1), axis=-1).squeeze(axis=-1)
    
    mean_u = np.mean(uncert)
    np.random.seed(3) 
    ind=np.random.choice(np.arange(N), images_n)
    #ind = [1544, 1968,1972,75, 128, 431, 175, 350, 1879, 61]
   
    n = 0
    
    cMap = []
    for value, colour in zip([0, 1, 2],["Black", "Yellow", "Red"]): cMap.append((value/2., colour))
    customColourMap = LinearSegmentedColormap.from_list("custom", cMap)
    for i in ind:

        #X = truex[i,:,:,0]
        u = uncert[i,:,:]
        #X = np.squeeze(X)
        if Adversarial:
         M = masked[i,:,:]
        P = predict[i,:,:]
        L = truey[i,:,:]
        if Adversarial:
            plt.figure()
            plt.imshow(np.squeeze(truex[i,:,:]), 'gray', interpolation='none')
            plt.imshow(np.squeeze(adv[i,:,:]),'gray', interpolation='none', alpha= 0.8)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.savefig(path_full + '{}_Adversarial_noise.png'.format(i))
            plt.close()
        
        plt.figure(figsize=(10,10))
        plt.imshow(L, customColourMap, interpolation='none')
        plt.title("Ground truth Label") 
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(path_full + '{}_Label_image.png'.format(i))
        plt.close()


        plt.figure(figsize=(10,10))
        plt.imshow(P, customColourMap, interpolation='none')
        plt.title("Predicted Label") 
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(path_full + '{}_Predicted_image.png'.format(i))
        plt.close()

        plt.figure(figsize=(10,10))
        im= plt.imshow(u, cmap='winter_r', interpolation='nearest')
        plt.title("Uncertainty map") 
        add_colorbar(im)
        #plt.clim(m_un,mx_un)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(path_full +'{}_uncertainty_heatmap.png'.format(i))
        plt.close()

        if Adversarial:
         if targeted:
            plt.figure(figsize=(10,10))
            plt.imshow(M, customColourMap, interpolation='none')
            plt.title("Masked Label") 
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.savefig(path_full + '{}_Masked_Label_image.png'.format(i))
            plt.close()

    
    # creating uncertainty file to record the predictive variance for 3 binary tasks:
    ##task 1: Everything vs Anterior

    mask1ant  = np.ma.masked_where(predict2 != 1, uncert) # masking all non-anterior
    anterior_unc = np.mean(mask1ant)
    mask2ant  = np.ma.masked_where(predict2 == 1, uncert) # masking  anterior 
    no_ant_unc = np.mean(mask2ant)
    ##task 2: Everything vs Posterior

    mask1po  = np.ma.masked_where(predict2 != 2, uncert) # masking all non-posterior
    posterior_unc = np.mean(mask1po)
    mask2po  = np.ma.masked_where(predict2 == 2, uncert) # masking posterior
    no_po_unc = np.mean(mask2po)



    textfile = open(path + 'Predictive_variance_tasks.txt','w')
    textfile.write('\n Average Predictive variance : ' +str(mean_u)) 
    textfile.write("\n---------------------------------")
    textfile.write('\n Predictive variance for all  anterior structures : ' +str(anterior_unc))         
    textfile.write('\n Predictive variance for non-anterior structures : ' +str(no_ant_unc)) 
    
    textfile.write("\n---------------------------------")
    textfile.write('\n Predictive variance for posterior portion : ' +str(posterior_unc))         
    textfile.write('\n Predictive variance for non-posterior structures : ' +str(no_po_unc)) 
    
    
    textfile.close()
    print('done saving uncertainty and creating images')
    return mean_u,  anterior_unc, posterior_unc



######################################################################
  
###############################################################################
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%

def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
############################################
# function to compute log base 10 - used for SNR
def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator
############################################
# actually computing Sensitivity
def sensitivity(y_true, y_pred):
    """Computes the sensitivity = Recall
    """
    TP = np.multiply(y_true, y_pred) #true positives
    num = np.sum(TP, axis=(1,2)) # numerator = counting the number of TP
    den = np.sum(y_true, axis=(1,2)) #denominator = TP + FN
    x = np.divide(num,den)
    ratio = x[np.logical_not(np.isnan(x))]
    mean_r = np.mean(ratio)
    return  mean_r
#actually computing 
def precision(y_true, y_pred):
    """ Conputes the Precision"""
    TP = np.multiply(y_true, y_pred) #true positives
    num = np.sum(TP, axis=(1,2)) # numerator = counting the number of TP
    den = np.sum(y_pred, axis=(1,2)) #denominator = TP + FP
    x = np.divide(num,den)
    ratio = x[np.logical_not(np.isnan(x))]
    mean_r = np.mean(ratio)
    return  mean_r


def dice(y_true, y_pred):
    true_mask, pred_mask = y_true, y_pred
    A=np.sum(true_mask, axis=(1,2))
    B=np.sum(pred_mask, axis=(1,2))
    im_sum = A+ B

    # Compute Dice coefficient
    intersection = np.multiply(true_mask, pred_mask)
    intersection_sum = np.sum(intersection, axis=(1,2))
    c=2. * intersection_sum / im_sum
    c_masked = np.ma.masked_invalid(c)
    c_masked_avr =np.mean(c_masked)
    var = np.var(c_masked)
    ###NEW PART ADDED ON 07-23-21
    ###WANT TO SAVE ALL DICE SCORES TO GIVE STD IN PAPER
    #print('dice', c_masked_avr)
    return c_masked_avr,var

def compute_H(mask_1a,mask_1b): 
    h = 0
    N = mask_1a.shape[0] # batch size
    for i in range(N):
        h_i=max(directed_hausdorff(mask_1b[i,:,:],mask_1a[i,:,:])[0],directed_hausdorff(mask_1a[i,:,:],mask_1b[i,:,:])[0]) # getting H-measure for each image in batch
        h+=h_i
    return h/N

# specificity
def specificity(y_true, y_pred):
    """ Conputes the specificity """

    TN = np.zeros(y_true.shape) #true negatives
    mas1= np.ma.masked_where( (y_true==0) & (y_pred==0), TN) 
    ss = np.ma.filled(mas1, fill_value=1)
    num = np.sum(ss, axis=(1,2)) # numerator = counting the number of TN
    s2 =np.zeros(y_true.shape)
    y =  np.ma.masked_where(  y_true==0, s2) 
    ss2 = np.ma.filled(y, fill_value=1)
    den = np.sum(ss2, axis=(1,2)) #denominator = TN + FP
    x = np.divide(num,den)
    ratio = x[np.logical_not(np.isnan(x))]
    mean_r = np.mean(ratio)
    return  mean_r  

def mask_anterior(y_true, y_pred):
    A,B = y_true, y_pred.numpy()
    mask1a  = np.ma.masked_where(A ==2, A) # masking all posterior 
    mask_1a=np.ma.filled(mask1a, fill_value=0)
    mask1b  = np.ma.masked_where(B ==2, B) # masking all posterior
    mask_1b=np.ma.filled(mask1b, fill_value=0)
    di,all_di=dice(mask_1a,mask_1b)
    hus = compute_H(mask_1a,mask_1b)
    sens = sensitivity(mask_1a,mask_1b)
    prec = precision(mask_1a,mask_1b)
    spec = specificity(mask_1a,mask_1b)
    rvd = RVD(mask_1a,mask_1b)
    os, us = Os_and_Us(mask_1a,mask_1b)
    return di,all_di, hus#, sens, prec, spec, rvd, os, us
def mask_posterior(y_true, y_pred):
    A,B = y_true, y_pred.numpy()
    mask1a  = np.ma.masked_where(A <2, A) # masking anterior
    mask_1a=np.ma.filled(mask1a, fill_value=0)
    mask2a  = np.ma.masked_where(mask_1a==2, mask_1a) # masking all posterior structures to 1
    mask_2a=np.ma.filled(mask2a, fill_value=1)

    mask1b  = np.ma.masked_where(B <2, B) # masking edema to 0
    mask_1b=np.ma.filled(mask1b, fill_value=0)
    mask2b  = np.ma.masked_where(mask_1b==2, mask_1b) # masking all remaining tumor structures to 1
    mask_2b=np.ma.filled(mask2b, fill_value=1)
    di,all_di=dice(mask_2a,mask_2b)
    hus = compute_H(mask_2a,mask_2b)
    sens = sensitivity(mask_2a,mask_2b)
    prec = precision(mask_2a,mask_2b)
    spec = specificity(mask_2a,mask_2b)
    rvd = RVD(mask_2a,mask_2b)
    os, us = Os_and_Us(mask_2a,mask_2b)
    return di,all_di, hus#, sens, prec, spec, rvd, os, us
def c_score(p,q ):
    d = 2*p*(1-q)/(p+(1-q))+ 2*(1-p)*q/((1-p)+q)
    if p <q:
        c = - d
    else:
        c = d
    return c

def RVD(true_mask, pred_mask ):   
    A=np.sum(true_mask, axis=(1,2))
    B=np.sum(pred_mask, axis=(1,2))
    diff = B - A
    r = diff / A
    r_masked = np.ma.masked_invalid(r)
    r_masked_avr =np.mean(r_masked)
    return r_masked_avr

def Os_and_Us(true_mask, pred_mask ):
    A=np.sum(true_mask, axis=(1,2))
    B=np.sum(pred_mask, axis=(1,2))

    intersection = np.multiply(true_mask, pred_mask)
    intersection_sum = np.sum(intersection, axis=(1,2))
    im_sum = A+ B - intersection_sum

    Over = np.mean(np.ma.masked_invalid((B - intersection_sum)/ im_sum))
    Under = np.mean(np.ma.masked_invalid((A - intersection_sum)/ im_sum))
    
    return Over, Under
######################################################################
##################################################################################
# Function to crop tensor x1 according to shape given by tensor x2
def crop_tensor(x1,x2):
    with tf.name_scope("crop"):
        x1_shape = tf.shape(x1) # tensor coming from encoder path
        x2_shape = tf.shape(x2) # corresponding tensor from decoder
        # offsets for the top left corner of the crop:
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return x1_crop
#########################################################################

def crop_to_wanted_shape(x, shape):
    """ function to crop input image to wanted shape:
    _______________________________________________________
    inputs: x = tensor to crop, shape = choosen dimension"""
    
    x1_shape = tf.shape(x) # tensor
    #x2_shape = tf.constant([x1_shape[0], shape,shape,x1_shape[-1]  ]) 
    # offsets for the top left corner of the crop:
    offsets = [0, (x1_shape[1] - shape) // 2, (x1_shape[2] - shape) // 2, 0]
    size = [-1,shape, shape, -1]
    x1_crop = tf.slice(x, offsets, size)
    return x1_crop

def crop_numpy_image(x,shape, num_ch = 3):
    """ function to crop input image to wanted shape when images are fed as numpy arrays:
    _______________________________________________________
    inputs: x = tensor to crop, shape = choosen dimension"""
    or_size = x.shape[1]
    start =(or_size - shape)/2 # starting point to slice image
    start = int(start)
    end_p = or_size - start
    end_p = int(end_p)
    
    if num_ch==3:
        im = x[:,start:end_p, start:end_p,:]
    else:
        im = x[:,start:end_p, start:end_p]
    return im