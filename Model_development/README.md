# Model Development

This folder contains all scripts used for building, training, tuning, and evaluating predictive models at both the **zone** and **regional** levels. It includes shared utilities, classical baselines, tree-based methods, and deep learning architectures, along with plotting code for diagnostics and visualization.

## Contents

- **Common Code (`1.Common_Code/`)**  
  - Provides utility functions
  - Shared preprocessing routines and reusable functions  
  - Service modules for zone- and regional-level data access  

- **Linear Regression (`2.Linear_Regression/`)**  
  - *Zone-Level and Regional-Level pipelines*  
  - Preprocessing scripts for climate and fire variables  
  - Model building and training routines  
  - Hyperparameter tuning with parameter grids  
  - Evaluation metrics and error analysis  
  - Selection of best-performing linear models  

- **XGBoost (`3.XGBoost/`)**  
  - Implements gradient-boosted tree models  
  - Zone-level and regional-level workflows  
  - Includes preprocessing, training, and tuning scripts  
  - Model evaluation and feature importance analysis  
  - Stores outputs and selected models  

- **Gaussian Process Regression (`4.GPR/`)**  
  - Probabilistic regression with uncertainty quantification  
  - Zone-level and regional-level pipelines  
  - Data preprocessing and kernel selection  
  - Hyperparameter optimization and evaluation  
  - Selection of final GPR models  

- **LSTM (`5.LSTM/`)**  
  - Sequence modeling for climate–fire time series  
  - Preprocessing with sliding windows and scaling  
  - Model architecture definition and training loops  
  - Hyperparameter tuning (window size, hidden units, learning rates)  
  - Evaluation of predictive accuracy  
  - Selection of best-performing LSTM models  

- **Transformer (`6.Transformer/`)**  
  - Attention-based sequence models for wildfire prediction  
  - Zone-level and regional-level implementations  
  - Preprocessing of sequential input features  
  - Model construction, training, and tuning  
  - Performance evaluation and model selection  

- **Plots (`7.Plots/`)**  
  - Code for generating visualization outputs  
  - Observed vs. predicted plots for all models and scopes  
  - Residual plots
  - Figures organized by model and zone/regional level  
