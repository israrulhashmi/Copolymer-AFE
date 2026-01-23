#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:36:50 2024

@author: Prof. DR. Tarak
"""
# Link for papers
#Ref 1.  J Polym Sci. 62, 1175 (2024);  https://doi.org/10.1002/pol.20230714

#Ref 2.  ACS Engineering Au 4, 91 (2024); https://doi.org/10.1021/acsengineeringau.3c00055
#IMPORT ALL LIBRARIES
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('/home/tarak/hashmi/shubham/tg_raw.csv') # Replace with your file path
smiles = data['SMILES']
target = data['tg']

# Convert SMILES to molecular fingerprints using RDKit
def smiles_to_fingerprint(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return np.zeros((256,), dtype=int)  # Return an empty fingerprint if invalid SMILES
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256), dtype=int)
#%%
# Apply the fingerprint conversion
X = np.array([smiles_to_fingerprint(sm) for sm in smiles])
y = target.values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the DNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256,)),  # Input layer matching fingerprint size
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])
#%%
# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=1)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate MSE and R² for training and testing sets
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print results
print(f'MSE (Training): {mse_train:.4f}')
print(f'R² (Training): {r2_train:.4f}')
print(f'MSE (Testing): {mse_test:.4f}')
print(f'R² (Testing): {r2_test:.4f}')
#%%
# Total data points
total_data_points = len(smiles)
data_points_used_for_training = len(X_train)
data_points_used_for_testing = len(X_test)

#%% Print the total and split data points
print(f'Total Data Points: {total_data_points}')
print(f'Number of Data Points Used for Training: {data_points_used_for_training}')
print(f'Number of Data Points Used for Testing: {data_points_used_for_testing}')

#%%

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
    plt.xlabel('Actual Tg Values (Train)', fontsize=14)
    plt.ylabel('Predicted Tg Values (Train)', fontsize=14)
    plt.title('Parity Plot - Training Data', fontsize=16)

    # Customize axis ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.colorbar(label='Actual Values (Train)', orientation='vertical')
    plt.show()

# Function to plot parity plot for test data points
def plot_parity_test():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    # Create scatter plot using plasma colormap
    plt.scatter(y_test, y_pred_test, c=y_test, cmap='plasma', s=80, alpha=0.8, edgecolor='k')
    
    # Add a diagonal line (perfect prediction line)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=2)

    # Labels and titles
    plt.xlabel('Actual Tg Values (Test)', fontsize=14)
    plt.ylabel('Predicted Tg Values (Test)', fontsize=14)
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

