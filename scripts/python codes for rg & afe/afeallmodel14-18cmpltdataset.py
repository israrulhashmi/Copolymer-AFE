

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:32:04 2023

@author: tarak
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Set random seed
np.random.seed(42)

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictorsafe.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targetsafe.txt', delimiter=',')

# Filter data for testing (values less than 14 or greater than 18)
test_indices = np.where((target_data < 14) | (target_data > 18))
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Define different training data point counts
training_point_counts = [100, 200, 300, 500, 1000, 2000, 3000, 5000, 7000, 8438]

# Initialize lists to store R2 for each model
r2_xgboost_list = []
r2_random_forest_list = []
r2_dnn_list = []

# Iterate over different training point counts
for num_points in training_point_counts:
    # Filter data for training (values between 14 and 18)
    train_indices = np.where((target_data >= 14) & (target_data <= 18))[0][:num_points]
    X_train = predictor_data[train_indices]
    y_train = target_data[train_indices]

    # Standardize the data for DNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the XGBoost regression model
    xgb_model = XGBRegressor(n_estimators=500)
    xgb_model.fit(X_train, y_train)
    y_pred_xgboost = xgb_model.predict(X_test)
    r2_xgboost = r2_score(y_test, y_pred_xgboost)
    r2_xgboost_list.append(r2_xgboost)
    
    print(f'R2 (Testing) for XGBoost with {num_points} training points: {r2_xgboost:.4f}')

    # Create and train the Random Forest regression model
    rf_model = RandomForestRegressor(n_estimators=500)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    r2_random_forest_list.append(r2_rf)
    
    print(f'R2 (Testing) for Random Forest with {num_points} training points: {r2_rf:.4f}')

    # Create and train the DNN model
    dnn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.01)
    dnn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    history = dnn_model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, verbose=0)
    y_pred_dnn = dnn_model.predict(X_test_scaled).flatten()
    r2_dnn = r2_score(y_test, y_pred_dnn)
    r2_dnn_list.append(r2_dnn)

    print(f'R2 (Testing) for DNN with {num_points} training points: {r2_dnn:.4f}')

# Plot the R2 for testing data for XGBoost, Random Forest, and DNN
plt.figure(figsize=(10, 6))
plt.plot(training_point_counts, r2_xgboost_list, label='XGBoost', marker='o')
plt.plot(training_point_counts, r2_random_forest_list, label='Random Forest', marker='o')
plt.plot(training_point_counts, r2_dnn_list, label='DNN', marker='o')
plt.xlabel('Training Data Point Count')
plt.ylabel('R2 (Testing)')
plt.title('Comparison of R2 (Testing) for XGBoost, Random Forest, and DNN')
plt.legend()
plt.grid(True)
plt.show()
