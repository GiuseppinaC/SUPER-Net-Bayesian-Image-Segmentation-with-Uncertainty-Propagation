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

├── brats_functions.py # Helper functions for BraTS (plots, metrics, preprocessing)
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
By default, you may need to update the data path, results path in each script:

In brats.py, modify line XX to point to your BraTS dataset folder.

In hippocampus.py, modify line YY.

In lungs.py, modify line ZZ.

(We will provide comments in each file to indicate where these changes are needed.)

**Training:**

If Training only:


```sh
python Brats.py 
```
(Similarly for the other datasets: hippocampus.py, lungs.py).
**Testing - Clean Test Data & Noisy Data:**

If Testing (with default options):


```sh
python Brats.py (change lines ...)
```



**Testing - Adversarial Attacks:**

When Testing with added Gaussian noise, we need to specify the level of noise (variance) :



When Testing with added Adversarial (FGSM) attacks, we need to specify:
- the level of the attack (epsilon)
- if targeted or not (default is untargeted)
- the fooling class (for targeted attacks)
