#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:25:00 2024

@author: Prof. DR. Tarak
"""
#LINK FOR PAPERS
# Ref 1: https://arxiv.org/abs/2409.09691
# Ref 2: DOI: 10.1039/D3CP03100D (Paper) Phys. Chem. Chem. Phys., 2023, 25, 25166-25176

#IMPORT ALL LIBRARIES
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#%% 
np.random.seed(42)
tf.random.set_seed(42)

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/afedata/sequence.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/afedata/property.txt', delimiter=',')

# Split the data into training and testing sets using an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(predictor_data, target_data, test_size=0.2, random_state=42)

#%% Create a simple DNN model using TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),      # Fully connected hidden layer with ReLU activation
    tf.keras.layers.Dense(1)                           # Output layer (1 neuron for regression)
])

# Compile the model with a learning rate of 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

# Train the DNN model for 100 epochs (you can change the number of epochs as needed)
epochs = 100
history = model.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=1)

#%% Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R2) score for both datasets
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print the R2 scores
print(f'R2 Score (Training): {r2_train:.4f}')
print(f'R2 Score (Testing): {r2_test:.4f}')

rmse_test = np.sqrt(mse_test)

# Print the RMSE on the test set
print(f'RMSE on Test Set: {rmse_test:.4f}')


# Total data points
total_data_points = len(predictor_data)
data_points_used_for_training = len(X_train)
data_points_used_for_testing = len(X_test)

#%% Print the total and split data points
print(f'Total Data Points: {total_data_points}')
print(f'Number of Data Points Used for Training: {data_points_used_for_training}')
print(f'Number of Data Points Used for Testing: {data_points_used_for_testing}')

#%% Creating parity plots for both training and test datasets




# Function to plot parity plot for training data points
def plot_parity_train():
    plt.figure(figsize=(6, 6))
    
    # Creating scatter plot using plasma colormap
    plt.scatter(y_train, y_pred_train, c=y_train, cmap='plasma', s=80, alpha=0.8, edgecolor='k')
    
    # Add a diagonal line (perfect prediction line)
    min_val = min(y_train.min(), y_pred_train.min())
    max_val = max(y_train.max(), y_pred_train.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=2)

    # Labels and titles
    plt.xlabel('Actual Free Energy Values (Train)', fontsize=14)
    plt.ylabel('Predicted Free Energy Values (Train)', fontsize=14)
    plt.title('Parity Plot - Training Data', fontsize=16)

    # Customize axis ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.colorbar(label='Actual Values (Train)', orientation='vertical')
    plt.show()

# Function to plot parity plot for test data points
def plot_parity_test():
    plt.figure(figsize=(6, 6))
    # Create scatter plot using plasma colormap
    plt.scatter(y_test, y_pred_test, c=y_test, cmap='plasma', s=80, alpha=0.8, edgecolor='k')
    
    # Add a diagonal line (perfect prediction line)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=2)

    # Labels and titles
    plt.xlabel('Actual Free Energy Values (Test)', fontsize=14)
    plt.ylabel('Predicted Free Energy Values (Test)', fontsize=14)
    plt.title('Parity Plot - Test Data', fontsize=16)

    # Customize axis ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.colorbar(label='Actual Values (Test)', orientation='vertical')
    plt.show()

# Call the functions to plot parity plots for train and test data
plot_parity_train()
plot_parity_test()


#Thankyou!!!!





