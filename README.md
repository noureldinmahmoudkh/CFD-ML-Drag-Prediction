# CFD-ML-Drag-Prediction: Data-Driven Modeling of Flow Behavior

## Overview
This project investigates the application of Machine Learning (ML) techniques to fluid dynamics problems, focusing on predicting the drag coefficient ($C_d$) as a function of the Reynolds number ($Re$) for flow over a cylinder.

A CFD-inspired empirical correlation was used to generate baseline data, and synthetic Gaussian noise was added to emulate experimental uncertainties (e.g., wind tunnel measurements). The resulting dataset was used to train and evaluate different machine learning models.

## Key Objectives
* Generate synthetic fluid dynamics data with realistic noise characteristics.
* Model the nonlinear relationship between Reynolds number and drag coefficient.
* Compare Polynomial Regression and Random Forest models for predictive performance.
* Evaluate models using Mean Squared Error (MSE) and $R^2$ metrics.

## Tech Stack
* **Python 3**
* **NumPy & Pandas** — data generation and processing
* **Scikit-Learn** — machine learning models and preprocessing
* **Matplotlib** — visualization and model comparison

## Project Structure
```text
├── data/                   
│   └── cfd_dataset.csv     # Generated CFD-inspired dataset with noise
├── src/                    
│   ├── cfd_data.py         # Data generation script
│   └── ml_model.py         # Model training, evaluation, and visualization
├── results/                
│   └── model_plot.png      # Model predictions vs data
├── .gitignore              
├── requirements.txt        
└── README.md
