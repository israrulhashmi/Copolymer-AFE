#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:11:45 2023

@author: tarak
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')

# Filter data for training (values between 3 and 5)
train_indices = np.where((target_data >= 3.0) & (target_data <= 5))[0][:1000]
X_train = predictor_data[train_indices]
y_train = target_data[train_indices]

# Filter data for testing (values outside the range 3 to 5)
test_indices = np.where((target_data < 3) | (target_data > 5))[0]
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Define the Random Forest model
rf_model = RandomForestRegressor(random_state=10)

# Define hyperparameters for the random search
param_grid = {
    'n_estimators': [1000],  # Number of trees in the forest
    'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
    'max_depth': [10],  # Maximum depth of the tree
    'min_samples_split': [10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [6],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]  # Method of selecting samples for training each tree
}

# Perform random search with 5-fold cross-validation
random_search = RandomizedSearchCV(rf_model, param_distributions=param_grid, n_iter=20, scoring='r2', cv=5, verbose=1, random_state=10)

# Fit the random search to the training data
random_search.fit(X_train, y_train)

# Get the best model from the random search
best_rf_model = random_search.best_estimator_

# Make predictions
y_pred_train = best_rf_model.predict(X_train)
y_pred_test = best_rf_model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R2) score for both datasets
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print the R2 scores
print(f'R2 Score (Training): {r2_train:.4f}')
print(f'R2 Score (Testing): {r2_test:.4f}')

# Print the best hyperparameters found
print("Best Hyperparameters:")
print(random_search.best_params_)

# Rest of the code for plotting and visualization remains the same...
