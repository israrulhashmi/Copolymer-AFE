import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')

# Initialize lists to store R2 scores
r2_scores_xgboost = []
r2_scores_randomforest = []
r2_scores_dnn = []

# Ranges for training
training_ranges = [(3, 3.5)]

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

    # Create the XGBoost regression model
    xgb_model = XGBRegressor(n_estimators=10000, learning_rate=0.1, max_depth=3, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgboost = xgb_model.predict(X_test)
    r2_xgboost = r2_score(y_test, y_pred_xgboost)
    r2_scores_xgboost.append(r2_xgboost)

    # Create the Random Forest regression model
    rf_model = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_randomforest = rf_model.predict(X_test)
    r2_randomforest = r2_score(y_test, y_pred_randomforest)
    r2_scores_randomforest.append(r2_randomforest)

    # Create and train the DNN model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
        Dense(32, activation='relu'), 
        Dense(32, activation='relu'), # Fully connected hidden layer with ReLU activation
        Dense(1)  # Output layer (1 neuron for regression)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    y_pred_dnn = model.predict(X_test)
    r2_dnn = r2_score(y_test, y_pred_dnn)
    r2_scores_dnn.append(r2_dnn)

    # Print R2 test scores for each range and model
    print(f'R2 Score (Testing) for {min_range} to {max_range} range:')
    print(f'  XGBoost: {r2_xgboost:.4f}')
    print(f'  Random Forest: {r2_randomforest:.4f}')
    print(f'  DNN: {r2_dnn:.4f}')
    print('')

# Create a comparison plot of R2 scores vs. Training Range
plt.figure(figsize=(10, 6))

# Plot R2 scores for XGBoost
plt.plot([f'{min_range}-{max_range}' for min_range, max_range in training_ranges], r2_scores_xgboost, marker='o', label='XGBoost')

# Plot R2 scores for Random Forest
plt.plot([f'{min_range}-{max_range}' for min_range, max_range in training_ranges], r2_scores_randomforest, marker='x', label='Random Forest')

# Plot R2 scores for DNN
plt.plot([f'{min_range}-{max_range}' for min_range, max_range in training_ranges], r2_scores_dnn, marker='s', label='DNN')

plt.title('R2 Score vs. Training Range')
plt.xlabel('Training Range', fontsize=15)
plt.ylabel('R2 Score', fontsize=15)
plt.xticks(rotation=45)
plt.legend()
plt.grid(False)

plt.tight_layout()
plt.show()
