# SUPER-Net-Bayesian-Image-Segmentation-with-Uncertainty-Propagation


This repository contains the implementation and pretrained models for **SUPER-Net**, a Bayesian framework for trustworthy image segmentation via uncertainty propagation.  

> **Paper abstract (snippet):**  
> We propose **SUPER-Net**, a Bayesian framework for trustworthy image segmentation via uncertainty propagation. Using Taylor series approximations, SUPER-Net propagates the mean and covariance of the model’s posterior distribution across nonlinear layers. It generates two outputs simultaneously: the segmented image and a pixel-wise uncertainty map, eliminating the need for expensive Monte Carlo sampling. SUPER-Net’s performance is extensively evaluated on MRI and CT scans under various noisy and adversarial conditions.

---

## Repository Structure

SUPER U-Net
│
├── brats.py # Main script for the BraTS dataset
├── hippocampus.py # Main script for the Hippocampus dataset
├── lungs.py # Main script for the Lungs dataset
│
├── brats_functions.py # Helper functions for BraTS (plots, metrics, preprocessing)
├── hippocampus_functions.py
├── lungs_functions.py
│
├── models/ # Folder containing pretrained models
│ ├── brats_model
│ ├── hippocampus_model
│ └── lungs_model
│
├── requirements.txt # Python dependencies
└── README.md # This file
