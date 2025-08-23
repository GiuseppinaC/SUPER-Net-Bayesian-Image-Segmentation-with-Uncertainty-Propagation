# SUPER-Net-Bayesian-Image-Segmentation-with-Uncertainty-Propagation


This repository contains the implementation and pretrained models for **SUPER-Net**, a Bayesian framework for trustworthy image segmentation via uncertainty propagation.  

> **Paper abstract (snippet):**  
> We propose **SUPER-Net**, a Bayesian framework for trustworthy image segmentation via uncertainty propagation. Using Taylor series approximations, SUPER-Net propagates the mean and covariance of the model’s posterior distribution across nonlinear layers. It generates two outputs simultaneously: the segmented image and a pixel-wise uncertainty map, eliminating the need for expensive Monte Carlo sampling. SUPER-Net’s performance is extensively evaluated on MRI and CT scans under various noisy and adversarial conditions.

---

## Repository Structure
```sh



├── brats.py # Main script for the BraTS dataset
├── hippocampus.py # Main script for the Hippocampus dataset
├── lungs.py # Main script for the Lungs dataset

├── Brats_functions.py # Helper functions for BraTS (plots, metrics, preprocessing)
├── hippocampus_functions.py
├── lungs_functions.py

├── models/ # Folder containing pretrained models
   ├── brats_model
   ├── hippocampus_model
   └── lungs_model

├── requirements.txt # Python dependencies
└── README.md # This file
```


## Requirements

Install all required python dependencies:

```sh
pip install -r requirements.txt
```

## Running Models
- **`main()`**  
  Used for model training. It also supports training adversarial attacks and testing.

- **`testing()`**  
  Used for evaluating the model. Supports both clean (noise-free) evaluation and testing with different levels of noise.


```sh
python Brats.py 
```
(Similarly for the other datasets: hippocampus.py, lungs.py).
 

**Model Saving**
By default, the code will automatically create a folder to store trained models:

```python
PATH = './Dataset/saved_models_SUPER_u-Net/epoch_{}/'.format(epochs)
# Modify as needed - Datasets will be Brats, Hippocampus and Lungs
```


