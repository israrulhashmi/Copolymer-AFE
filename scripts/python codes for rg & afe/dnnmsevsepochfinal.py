#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:53:12 2024

@author: tarak
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictorsafe.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targetsafe.txt', delimiter=',')

np.random.seed(42)
tf.random.set_seed(42)

# Define different numbers of data points for training
data_points = [100, 1000, 8436]
markers = ['o', '^', 's']
colors = ['b', 'g', 'r']

plt.figure(figsize=(6, 6))

for idx, num_points in enumerate(data_points):
    # Filter data for training based on the number of data points
    selected_indices = np.random.choice(len(target_data), num_points, replace=False)
    X_train = predictor_data[selected_indices]
    y_train = target_data[selected_indices]

    # Create a simple DNN model using TensorFlow/Keras
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),  
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model with a learning rate of 0.01
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

    # Train the DNN model for 150 epochs
    epochs = 150
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=1000, verbose=0)

    # Extract the MSE values for training
    train_mse = history.history['mse']

    # Plot the MSE vs Epochs for each case
    plt.plot(range(1, epochs + 1), train_mse, color=colors[idx], label=f'Train MSE ({num_points} data points)', linewidth=2)
    
    # Plot markers at each 20 epochs
    for i in range(10, epochs + 1, 20):
        plt.plot(i, train_mse[i-1], marker=markers[idx], color=colors[idx], markersize=8)

# Create a custom legend with marker styles
legend_labels = ['N=100', 'N=1000', 'N=8436']
legend_markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=14, linestyle='None'),
                  plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='g', markersize=14, linestyle='None'),
                  plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='r', markersize=14, linestyle='None')]
plt.legend(legend_markers, legend_labels, loc='upper right', fontsize=12,frameon=False)

plt.xlabel('Epochs', fontsize=14, fontweight='bold')
plt.ylabel('MSE', fontsize=14, fontweight='bold')
#plt.title('MSE vs Epochs (Training) for Different Data Points (DNN)', fontsize=16, fontweight='bold')

# Set the y-axis ticks
plt.yticks([0, 20, 40, 60, 80],fontsize=12,fontweight='bold')

# Set the x-axis ticks
plt.xticks([0, 50, 100, 150],fontsize=12,fontweight='bold')

# Set the y-axis limits to show from 0 to 80
plt.ylim(-1, 80)

# Set the tick labels to bold
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.weight'] = 'bold'

# Set the plot border to be thick
plt.rcParams['axes.linewidth'] = 3

plt.show()