#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:27:32 2023

@author: tarak
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:06:25 2023

@author: tarak
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')

# Initialize a list to store R2 scores for XGBoost
r2_scores_xgboost = []

# Ranges for training
training_ranges = [(3, 5)]

# Number of data points for training in each range
num_points_per_range = 4000

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10000],
    'learning_rate': [0.4],
    'max_depth': [3]
}

# Loop through different training ranges
for min_range, max_range in training_ranges:
    # Filter data for training (values within the specified range)
    train_indices = np.where((target_data >= min_range) & (target_data <= max_range))[0][:num_points_per_range]
    X_train = predictor_data[train_indices]
    y_train = target_data[train_indices]

    # Filter data for testing (values outside the specified range)
    test_indices = np.where((target_data < min_range) | (target_data > max_range))
    X_test = predictor_data[test_indices]
    y_test = target_data[test_indices]

    # Create the XGBoost regression model
    xgb_model = XGBRegressor(random_state=42)

    # Grid Search for hyperparameter optimization
    xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3)
    xgb_grid.fit(X_train, y_train)

    # Use the best estimator from Grid Search
    best_xgb_model = xgb_grid.best_estimator_

    # Fit the best model to the training data
    best_xgb_model.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred_xgboost = best_xgb_model.predict(X_test)
    r2_xgboost = r2_score(y_test, y_pred_xgboost)
    r2_scores_xgboost.append(r2_xgboost)

    # Print R2 test scores for XGBoost
    print(f'R2 Score (Testing) for {min_range} to {max_range} range (XGBoost): {r2_xgboost:.4f}')
    print(f'Best Hyperparameters: {xgb_grid.best_params_}')

# Create a comparison plot of R2 scores vs. Training Range for XGBoost
plt.figure(figsize=(10, 6))

# Plot R2 scores for XGBoost
plt.plot([f'{min_range}-{max_range}' for min_range, max_range in training_ranges], r2_scores_xgboost, marker='o', label='XGBoost')

plt.title('R2 Score vs. Training Range for XGBoost')
plt.xlabel('Training Range', fontsize=15)
plt.ylabel('R2 Score', fontsize=15)
plt.xticks(rotation=45)
plt.legend()
plt.grid(False)

plt.tight_layout()
plt.show()
