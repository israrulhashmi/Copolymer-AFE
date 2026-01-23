#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:27:48 2023

@author: tarak
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf  # Import TensorFlow

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')



np.random.seed(42)
tf.random.set_seed(42)
# Initialize a list to store R2 scores for DNN
r2_scores_dnn = []

# Ranges for training
training_ranges = [(3, 5)]

# Number of data points for training in each range
num_points_per_range = 4000

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

    # Create and train the DNN model with learning rate 0.1
    model = Sequential([
        Dense(100, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
       
        Dense(50, activation='relu'),
        Dense(50, activation='relu'),
        
        # Fully connected hidden layer with ReLU activation
        
        Dense(1)  # Output layer (1 neuron for regression)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    y_pred_dnn = model.predict(X_test)
    r2_dnn = r2_score(y_test, y_pred_dnn)
    r2_scores_dnn.append(r2_dnn)

    # Print R2 test scores for DNN
    print(f'R2 Score (Testing) for {min_range} to {max_range} range (DNN): {r2_dnn:.4f}')

    # ... (previous code) ...

    y_pred_dnn = model.predict(X_test)
    mse_dnn = mean_squared_error(y_test, y_pred_dnn)  # Calculate MSE
    r2_dnn = r2_score(y_test, y_pred_dnn)
    r2_scores_dnn.append(r2_dnn)

    # Print R2 and MSE test scores for DNN
    print(f'R2 Score (Testing) for {min_range} to {max_range} range (DNN): {r2_dnn:.4f}')
    print(f'Mean Squared Error (Testing) for {min_range} to {max_range} range (DNN): {mse_dnn:.4f}')

# ... (rest of the code) ...
