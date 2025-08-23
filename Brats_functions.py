
import tensorflow as tf
from tensorflow import keras
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

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
from scipy.spatial.distance import directed_hausdorff
from scipy import stats


plt.ioff()
from mpl_toolkits import axes_grid1
def plot_saliency_map(path,truex,truey, saliency1, saliency2, images_n):
    print('going over function uncertainty for adversarial')
     
    path_full= path 
    if not os.path.exists(path_full):
        os.makedirs(path_full)

    #sigma = softplus(sigma)
    N = truex.shape[0] 
    

    ind = [420, 419, 422, 423]
    n = 0
    
    cMap = []
    for value, colour in zip([0, 1, 2, 3, 4],["Black", "Cyan" ,"Lime","Yellow", "Red"]): cMap.append((value/4., colour))
    customColourMap = LinearSegmentedColormap.from_list("custom", cMap)
    for i in ind:
        L = truey[i,:,:]
        #x = truex[i,:,:,0]
        plt.figure(figsize=(60,10))
        for j in range(4):
            f = j+1
            plt.subplot(1,6,f)
            
            plt.imshow(np.squeeze(truex[i,:,:,0]), 'gray', interpolation='none' ,alpha= 0.5)
            plt.imshow(saliency1[i,:,:, j],cmap='RdBu_r', interpolation='nearest', alpha= 0.7)
            
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.title.set_text('Slice_{}'.format(f))
        plt.subplot(1,6,5)   
        plt.imshow(L, customColourMap, interpolation='none')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.title.set_text('Prediction')

        plt.subplot(1,6,6)   
        plt.imshow(np.squeeze(truex[i,:,:,0]), 'gray', interpolation='none')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.title.set_text('Flair')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        plt.savefig(path_full + '{}_Saliency_map1.png'.format(i))
        plt.close()


        plt.figure(figsize=(60,10))
        for j in range(4):
            f = j+1
            plt.subplot(1,6,f)
            
            plt.imshow(truex[i,:,:,0], 'gray', interpolation='none',  alpha= 0.5)
            plt.imshow(saliency2[i,:,:, j],cmap='gist_heat_r', interpolation='nearest', alpha= 0.7)
            
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.title.set_text('Slice_{}'.format(f))
        plt.subplot(1,6,5)   
        plt.imshow(L, customColourMap, interpolation='none')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.title.set_text('Prediction')
        plt.subplot(1,6,6)   
        plt.imshow(truex[i,:,:,0], 'gray', interpolation='none')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.title.set_text('Flair')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        plt.savefig(path_full + '{}_Saliency_map2.png'.format(i))
        plt.close()
        
        
        sal1 = np.mean(saliency1[i,:,:,:], axis =-1)
        sal2 = np.mean(saliency2[i,:,:,:], axis =-1)
        
        plt.figure(figsize=(10,10))
        plt.imshow(truex[i,:,:,0], 'gray', interpolation='none',  alpha= 0.5)
        im= plt.imshow(sal1,cmap='RdBu_r', interpolation='nearest', alpha= 0.7)
        plt.title("Saliency map 1") 
        add_colorbar(im)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(path_full + '{}_saliency_mean1.png'.format(i))
        plt.close()
        
        plt.figure(figsize=(10,10))
        plt.imshow(truex[i,:,:,0], 'gray', interpolation='none',  alpha= 0.5)
        im= plt.imshow(sal2,cmap='gist_heat_r', interpolation='nearest', alpha= 0.7)
        plt.title("Saliency map 2") 
        add_colorbar(im)
        #plt.clim(m_un,mx_un)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(path_full + '{}_saliency_mean2.png'.format(i))
        plt.close()
        

def get_mask(image, target, tumor =False):
    label = tf.argmax(image, axis=-1)
    if tumor:
        mask1  = label>0
    else:
        mask1  = label ==target 
    mask2 = tf.boolean_mask(image, mask1)
    #mask_mean = tf.math.reduce_mean(mask2)
    mask_mean = tf.math.reduce_sum(mask2)
    return mask_mean

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

def uncert_for_corr(logits,sigma, size):
    predict = np.argmax(logits, axis = -1)
    mask_uncert = np.ma.masked_invalid(sigma)
    uncert= np.take_along_axis(sigma, np.expand_dims(predict, axis=-1), axis=-1).squeeze(axis=-1)
    uncert2 = np.reshape(uncert, [size,-1])
    mean_u = np.mean(uncert2, axis =-1)
    
    mask_un_1 = np.ma.masked_where(predict == 0, uncert)
    mask_un_1 = np.reshape(mask_un_1, [size,-1])
    mean_tumor = np.mean(mask_un_1, axis =-1)

    mask1a  = np.ma.masked_where(predict ==2, predict) # masking edema to 0
    mask_1a=np.ma.filled(mask1a, fill_value=0)
    mask_core = np.ma.masked_where(mask_1a == 0, uncert)
    mask_core = np.reshape(mask_core, [size,-1])
    mean_core = np.mean(mask_core, axis =-1)
    
    mask1eh  = np.ma.masked_where(predict != 4, uncert) # masking all non-enhancing
    mask1eh = np.reshape(mask1eh, [size,-1])
    enh_unc = np.mean(mask1eh, axis =-1)
    return mean_tumor, mean_core, enh_unc, mean_u


def save_adversarial_uncertainty(path,truex,adv, logits, truey,sigma,masked, images_n, Adversarial = True, targeted=True):
    print('going over function uncertainty for adversarial')
     
    path_full= path + './test_images/'
    if not os.path.exists(path_full):
        os.makedirs(path_full)

    #sigma = softplus(sigma)
    N = truex.shape[0] 
    predict = np.argmax(logits, axis = -1)
    predict2 = predict
    mask_uncert = np.ma.masked_invalid(sigma)
    uncert= np.take_along_axis(sigma, np.expand_dims(predict, axis=-1), axis=-1).squeeze(axis=-1)
    
    mean_u = np.mean(uncert)

    np.random.seed(70) 
    ind=np.random.choice(np.arange(N), images_n)
    
    n = 0
    
    cMap = []
    for value, colour in zip([0, 1, 2, 3, 4],["Black", "Cyan" ,"Lime","Yellow", "Red"]): cMap.append((value/4., colour))
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
         plt.figure(figsize=(40,10))
         for j in range(4):
            f = j+1
            plt.subplot(1,4,f)
            plt.imshow(truex[i,:,:,j], 'gray', interpolation='none')
            plt.imshow(adv[i,:,:, j],'gray', interpolation='none', alpha= 0.9)
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
    # task 1: tumor vs everything else(background)
    mask_un_1 = np.ma.masked_where(predict2 == 0, uncert)
    mask_un_2 = np.ma.masked_where(predict2  >0, uncert)
    mean_background = np.mean(mask_un_2)
    mean_tumor = np.mean(mask_un_1)
    #task 2: core
    mask1a  = np.ma.masked_where(predict2 ==2, predict2) # masking edema to 0
    mask_1a=np.ma.filled(mask1a, fill_value=0)
    mask2a  = np.ma.masked_where(predict2>0, mask_1a) # masking all remaining tumor structures to 1
    mask_2a=np.ma.filled(mask2a, fill_value=1)
    mask_un_3 = np.ma.masked_where(mask_2a == 1, uncert) # true for core structures
   
    mask_core = np.ma.masked_where(mask_1a == 0, uncert)
    mean_core = np.mean(mask_core)
    mean_no_core = np.mean(mask_un_3)
    #task 3: enhancing
    mask1eh  = np.ma.masked_where(predict2 != 4, uncert) # masking all non-enhancing
    enh_unc = np.mean(mask1eh)
    mask2eh  = np.ma.masked_where(predict2 == 4, uncert) # masking  non-enhancing 
    no_enh_unc = np.mean(mask2eh)

    # save uncertainty per class:
    mask_class1  = np.ma.masked_where(predict2 != 1, uncert) # masking all non-class1 elements
    class1_unc = np.mean(mask_class1 ) # uncert for necrosis
    mask_class2  = np.ma.masked_where(predict2 != 2, uncert) # masking all non-class2 elements
    class2_unc = np.mean(mask_class2 ) # uncert for edema
    mask_class3  = np.ma.masked_where(predict2 != 3, uncert) # masking all non-class3 elements
    class3_unc = np.mean(mask_class3 ) # uncert for non-enhancing

    #averaging uncertainty for correct and incorrect predictions:
    mask_correct  = np.ma.masked_where(predict2 != truey, uncert) # masking all non-correct elements
    correct_unc = np.mean(mask_correct )
    mask_incorrect  = np.ma.masked_where(predict2 == truey, uncert) # masking all correct elements
    incorrect_unc = np.mean(mask_correct )

    textfile = open(path + 'Predictive_variance_tasks.txt','w')
    textfile.write('\n Average Predictive variance : ' +str(mean_u)) 
    textfile.write("\n---------------------------------")
    textfile.write('\n Predictive variance for all tumor structures : ' +str(mean_tumor))         
    textfile.write('\n Predictive variance for non-tumor structures : ' +str(mean_background)) 
    
    textfile.write("\n---------------------------------")
    textfile.write('\n Predictive variance for core portion : ' +str(mean_core))         
    textfile.write('\n Predictive variance for non-core structures : ' +str(mean_no_core)) 
    
    textfile.write("\n---------------------------------")
    textfile.write('\n Predictive variance for enhancing portion : ' +str(enh_unc))         
    textfile.write('\n Predictive variance for non-enhancing portion : ' +str(no_enh_unc)) 
    
    textfile.write("\n------------------Per Task---------------")
    textfile.write(str(mean_tumor))
    textfile.write(str(mean_core))
    textfile.write(str(enh_unc)) 
    textfile.write("\n-----------Uncertainty Per Class--------------")
    textfile.write('\n Predictive variance for class 0 - non-tumor : ' +str(mean_background)) 
    textfile.write('\n Predictive variance for class 1 - necrosis: ' +str(class1_unc)) 
    textfile.write('\n Predictive variance for class 2 - edema: ' +str(class2_unc)) 
    textfile.write('\n Predictive variance for class 3 - non-enhancing: ' +str(class3_unc)) 
    textfile.write('\n Predictive variance for class 4 - enhancing: ' +str(enh_unc))
    textfile.write("\n-------------------------")
    textfile.write(str(mean_background)) 
    textfile.write(str(class1_unc)) 
    textfile.write(str(class2_unc)) 
    textfile.write(str(class3_unc)) 
    textfile.write(str(enh_unc))
    textfile.write("\n-------------------------")
    textfile.write('\n Predictive variance for correct & Incorrect : ' ) 
    textfile.write(str(correct_unc)) 
    textfile.write(str(incorrect_unc)) 
    textfile.close()
    print('done saving uncertainty and creating images')
    return mean_u,  mean_background, mean_tumor, class1_unc, class2_unc, class3_unc, enh_unc



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
    #print(ratio, "  TP+FP - for precision")
    mean_r = np.mean(ratio)
    #print(mean_r, "  TP+FP - for precision")
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

    return c_masked_avr,c_masked

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

def mask_tumor(y_true, y_pred):
    A,B = y_true, y_pred.numpy()
    mask1a  = np.ma.masked_where(A > 0, A) # masking all tumor structures to 1
    mask_1a=np.ma.filled(mask1a, fill_value=1)
    mask1b  = np.ma.masked_where(B > 0, B) # masking all tumor structures to 1
    mask_1b=np.ma.filled(mask1b, fill_value=1)
    di,all_di=dice(mask_1a,mask_1b)
    hus = compute_H(mask_1a,mask_1b)
    sens = sensitivity(mask_1a,mask_1b)
    prec = precision(mask_1a,mask_1b)
    spec = specificity(mask_1a,mask_1b)
    return di,all_di, hus, sens, prec , spec
def mask_core(y_true, y_pred):
    A,B = y_true, y_pred.numpy()
    mask1a  = np.ma.masked_where(A ==2, A) # masking edema to 0
    mask_1a=np.ma.filled(mask1a, fill_value=0)
    mask2a  = np.ma.masked_where(mask_1a>0, mask_1a) # masking all remaining tumor structures to 1
    mask_2a=np.ma.filled(mask2a, fill_value=1)

    mask1b  = np.ma.masked_where(B ==2, B) # masking edema to 0
    mask_1b=np.ma.filled(mask1b, fill_value=0)
    mask2b  = np.ma.masked_where(mask_1b>0, mask_1b) # masking all remaining tumor structures to 1
    mask_2b=np.ma.filled(mask2b, fill_value=1)
    di,all_di=dice(mask_2a,mask_2b)
    hus = compute_H(mask_2a,mask_2b)
    sens = sensitivity(mask_2a,mask_2b)
    prec = precision(mask_2a,mask_2b)
    spec = specificity(mask_2a,mask_2b)
    return di,all_di, hus, sens, prec, spec
def mask_enh(y_true, y_pred):
    A,B = y_true, y_pred.numpy()  
    mask1a  = np.ma.masked_where(A != 4, A) # masking all non-enhancing to 0
    mask_1a=np.ma.filled(mask1a, fill_value=0)
    mask1a = np.ma.masked_where(A == 4, mask_1a) # mask enhancing tumor  to 1
    mask_1a=np.ma.filled(mask1a, fill_value=1)  
    mask1b  = np.ma.masked_where(B != 4, B)  # masking all non-enhancing to 0
    mask_1b=np.ma.filled(mask1b, fill_value=0)
    mask1b  = np.ma.masked_where(B == 4, mask_1b) # mask enhancing tumor to 1
    mask_1b=np.ma.filled(mask1b, fill_value=1)
    di,all_di=dice(mask_1a,mask_1b)
    hus = compute_H(mask_1a,mask_1b)
    sens = sensitivity(mask_1a,mask_1b)
    prec = precision(mask_1a,mask_1b)
    spec = specificity(mask_1a,mask_1b)
    return di,all_di, hus, sens, prec, spec
  

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

##############################################################################

def expand_to_shape(data, shape, border='CONSTANT'):
    """
    Expands the array to the given image shape by padding it with a border (expects a tensor of shape [batches, nx, ny, channels].
    :param data: the array to expand
    :param shape: the target shape
    :param border: default CONSTANT -> alternatives: "SYMMETRIC" , "REFLECT"(probably not useful)
    """
    diff_nx = shape[1] - data.shape[1]
    diff_ny = shape[2] - data.shape[2]
    #NEEDED IF LEFT & RIGHT (TOP & BOTTOM) ARE DIFFERENT
    offset_nx_left = diff_nx // 2
    offset_ny_left = diff_ny // 2
    offset_nx_right = diff_nx - offset_nx_left
    offset_ny_right = diff_ny - offset_ny_left
    paddings = tf.constant([[0, 0,], [offset_nx_left, offset_nx_right], [offset_ny_left,offset_ny_right], [0, 0,]])
    expanded = tf.pad(data,paddings, border )
    return expanded

# Function to load a single pickle
def load_pickle(filename):
    # filename comes as bytes from tf.py_function
    fname = filename.numpy().decode("utf-8")
    with open(fname, "rb") as f:
        x, y = pickle.load(f)  # x: (B, C, H, W), y: (B, H, W, num_classes)
    x = x.transpose(0, 2, 3, 1)  # to NHWC
    return x.astype("float32"), y.astype("float32")

def tf_load_pickle(filename):
    x, y = tf.py_function(func=load_pickle, inp=[filename], Tout=[tf.float32, tf.float32])
    # Ensure TensorFlow knows the shape
    x.set_shape([None, 204, 204, 4])
    y.set_shape([None, 204, 204])
    return x, y
    

def salt_and_pepper(image,  p, q = 0.5):
        '''
        p = ['amount'] - like noise variance
        q = ['salt_vs_pepper'] - ratio of salt and pepper
        '''
        #np.random.seed(seed)
        # Detect if a signed image was input
        if np.amin(image) < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.zeros(shape=image.shape)
        flipped = np.random.choice([True, False], size=image.shape,p=[p, 1 - p])
        salted = np.random.choice([True, False], size=image.shape,p=[q, 1 - q])
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = low_clip
        return out

def save_uncertainty(path,images_n, noise, where_noise ):
    print('going over function uncertainty')
    if noise == 0:
        #PATH = path
        file1 = open(path +"uncertainty_info.pkl", 'rb')
    else:
        if where_noise == "O":   
            
            file1 = open(path+ "uncertainty_info_on_object_noise_{}.pkl".format(noise,noise), 'rb')
        elif where_noise == "B":
          
            file1 = open(path +"uncertainty_info_on_background_noise_{}.pkl".format(noise), 'rb')
        else:
          
            file1 = open(path+"uncertainty_info_noise_{}.pkl".format(noise,noise), 'rb')

    logits, sigma, truex, truey, testacc =   pickle.load(file1)
    file1.close()  
    mean_u,  mean_background, mean_tumor, class1_unc, class2_unc, class3_unc, enh_unc = save_adversarial_uncertainty(path, truex, 0, logits, truey, sigma, 0,images_n,Adversarial=False )
    return mean_u,  mean_background, mean_tumor, class1_unc, class2_unc, class3_unc, enh_unc
