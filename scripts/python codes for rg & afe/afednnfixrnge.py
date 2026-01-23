#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:22:26 2023

@author: tarak
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Set random seed
np.random.seed(42)

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/afedata/sequence.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/afedata/property.txt', delimiter=',')

# Filter data for testing (values less than 14 or greater than 18)
test_indices = np.where((target_data < 14.0) | (target_data > 18))
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Define different training data point counts
training_point_counts = [8000]

# Initialize list to store R2 for DNN
r2_dnn_list = []

# Iterate over different training point counts
for num_points in training_point_counts:
    # Filter data for training (values between 14 and 18)
    train_indices = np.where((target_data >= 14) &  (target_data <= 18.0))[0][:num_points]
    X_train = predictor_data[train_indices]
    y_train = target_data[train_indices]
    # Generate predictions for the training data
# Standardize the data for DNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the DNN model
    dnn_model = Sequential([
        Dense(40, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(20, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.01)
    dnn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    history = dnn_model.fit(X_train_scaled, y_train, epochs=1000, batch_size=100, verbose=1)
    # Generate predictions for the training data
    y_pred_train = dnn_model.predict(X_train_scaled).flatten()

    y_pred_dnn = dnn_model.predict(X_test_scaled).flatten()
    r2_dnn = r2_score(y_test, y_pred_dnn)
    r2_dnn_list.append(r2_dnn)


#%%    

print(f'R2 (Testing) for DNN with {num_points} training points: {r2_dnn:.4f}')


#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(8.05, 8.13))

# Scatter plot for test data with black color, darkened markers and larger marker size
plt.scatter(y_test, y_pred_dnn, c='none', marker='v', edgecolors='red', alpha=0.9, label='Test Data', s=200)
plt.scatter(y_train, y_pred_train, c='none', marker='o', edgecolors='green', alpha=1, label='Train Data', s=160)

# Add a red diagonal line from one corner to the opposite corner
min_val = min(min(y_train), min(y_pred_dnn))
max_val = max(max(y_test), max(y_pred_dnn))
plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=4)

# Set the aspect ratio to 'equal' for a full square box
plt.axis('equal')

# Darken the outer square lines by setting spines properties
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(4)
    spine.set_color('black')

# Remove the outline of the legend box
legend = plt.legend( fontsize=18,frameon=False)
for label in legend.get_texts():
    label.set_fontweight('bold')
    
plt.xlabel('Actual Free Energy Values', fontsize=25)
plt.ylabel('Predicted Free Energy Values', fontsize=25)
plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])
plt.xticks([10, 15, 20, 25], fontsize=18, weight='bold')
plt.yticks([10, 15, 20, 25], fontsize=18, weight='bold')

plt.tight_layout()
plt.show()



