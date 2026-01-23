 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 19:40:20 2023

@author: hashmi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score

np.random.seed(42)

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictorsafe.txt', delimiter=',')
target_data = np.genfromtxt('/home/tarak/hashmi/targetsafe.txt', delimiter=',')

# Print the lowest and maximum values in target_data
lowest_value = np.min(target_data)
max_value = np.max(target_data)

print(f'Lowest target value: {lowest_value:.4f}')
print(f'Maximum target value: {max_value:.4f}')

# Function to train and evaluate the model for a specified data range
def train_and_evaluate(min_value, max_value):
    # Filter data for training within the specified range
    train_indices = np.where((target_data >= min_value) & (target_data <= max_value))[0][:300]
    X_train = predictor_data[train_indices]
    y_train = target_data[train_indices]

    # Test using data points outside the specified range
    test_indices = np.where((target_data < min_value) | (target_data > max_value))[0]
    X_test = predictor_data[test_indices]
    y_test = target_data[test_indices]

    # Train and evaluate using a RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_train_rf = rf_model.predict(X_train)
    y_pred_test_rf = rf_model.predict(X_test)
    r2_train_rf = r2_score(y_train, y_pred_train_rf)
    r2_test_rf = r2_score(y_test, y_pred_test_rf)

    print(f'R2 Score (Training) for {min_value} to {max_value} range (Random Forest): {r2_train_rf:.4f}')
    print(f'R2 Score (Testing) for {min_value} to {max_value} range (Random Forest): {r2_test_rf:.4f}')

    # Train and evaluate using XGBoost
    xgb_model = XGBRegressor(n_estimators=1000,learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_train_xgb = xgb_model.predict(X_train)
    y_pred_test_xgb = xgb_model.predict(X_test)
    r2_train_xgb = r2_score(y_train, y_pred_train_xgb)
    r2_test_xgb = r2_score(y_test, y_pred_test_xgb)

    print(f'R2 Score (Training) for {min_value} to {max_value} range (XGBoost): {r2_train_xgb:.4f}')
    print(f'R2 Score (Testing) for {min_value} to {max_value} range (XGBoost): {r2_test_xgb:.4f}')

    # Train and evaluate using DNN
    dnn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1,activation='linear')
    ])

    optimizer = Adam(learning_rate=0.02)
    dnn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    history = dnn_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0, verbose=0)

    y_pred_test_dnn = dnn_model.predict(X_test).flatten()
    r2_test_dnn = r2_score(y_test, y_pred_test_dnn)

    print(f'R2 Score (Testing) for {min_value} to {max_value} range (DNN): {r2_test_dnn:.4f}')
    return r2_test_rf, r2_test_xgb, r2_test_dnn

# Ranges to evaluate
ranges = [(14, 15), (14, 16), (14, 17), (14, 18)]

# Lists to store R2 scores for testing
r2_scores_rf = []
r2_scores_xgb = []
r2_scores_dnn = []

# Train and evaluate for each range
for min_val, max_val in ranges:
    r2_test_rf, r2_test_xgb, r2_test_dnn = train_and_evaluate(min_val, max_val)
    r2_scores_rf.append(r2_test_rf)
    r2_scores_xgb.append(r2_test_xgb)
    r2_scores_dnn.append(r2_test_dnn)

# Plot comparison of range vs R2 score (testing) for all models
plt.figure(figsize=(10, 6))
plt.plot([f'{min_val} to {max_val}' for min_val, max_val in ranges], r2_scores_rf, label='Random Forest', marker='o', color='pink')
plt.plot([f'{min_val} to {max_val}' for min_val, max_val in ranges], r2_scores_xgb, label='XGBoost', marker='o', color='gold')
plt.plot([f'{min_val} to {max_val}' for min_val, max_val in ranges], r2_scores_dnn, label='DNN', marker='o', color='cyan')
plt.xlabel('Data Range')
plt.ylabel('R2 Score (Testing)')
plt.title('Comparison of Range vs R2 Score (Testing) for Different Models')
plt.legend()
plt.grid(False)
plt.show()
