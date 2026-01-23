import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')

# Filter data for testing (values between 0 and 7)
test_indices = np.where((target_data >= 0) & (target_data <= 7))
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Initialize lists to store R2 scores
r2_scores_xgboost = []
r2_scores_randomforest = []
r2_scores_dnn = []
data_point_counts = []

# Loop through different training data point counts
for num_points in [200, 500, 1000, 2000, 3000, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 14366]:
    # Filter data for training (values between 3.5 and 4)
    train_indices = np.where((target_data >= 3.0) & (target_data <= 5))[0][:num_points]
    X_train = predictor_data[train_indices]
    y_train = target_data[train_indices]

    # Create the XGBoost regression model
    xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=5, random_state=42)
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
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
        tf.keras.layers.Dense(64, activation='relu'),      # Fully connected hidden layer with ReLU activation
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),# Fully connected hidden layer with ReLU activation
        tf.keras.layers.Dense(1)                           # Output layer (1 neuron for regression)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    y_pred_dnn = model.predict(X_test)
    r2_dnn = r2_score(y_test, y_pred_dnn)
    r2_scores_dnn.append(r2_dnn)

    data_point_counts.append(num_points)

# Create a comparison plot of R2 scores vs. Data Point Counts
plt.figure(figsize=(12, 6))

# Plot R2 scores for XGBoost
plt.plot(data_point_counts, r2_scores_xgboost, marker='o', linestyle='-', label='XGBoost R2')

# Plot R2 scores for Random Forest
plt.plot(data_point_counts, r2_scores_randomforest, marker='x', linestyle='-', label='Random Forest R2')

# Plot R2 scores for DNN
plt.plot(data_point_counts, r2_scores_dnn, marker='s', linestyle='-', label='DNN R2')

plt.title('R2 Score vs. Training Data Point Count for the range 3.0 to 5')
plt.xlabel('Training Data Point Count', fontsize=15)
plt.ylabel('R2 Score', fontsize=15)
plt.xticks(data_point_counts, fontsize=12, rotation=45)
plt.yticks(np.arange(-0.3, 1.1, 0.1), fontsize=12)  # Modified y-axis range
plt.ylim([-0.3, 1])  # Modified y-axis range
plt.legend(fontsize=12)
plt.grid(False)  # Remove grid

# Create parody plots for both training and test datasets for the DNN model
plt.figure(figsize=(12, 5))

# Parody plot for training data
plt.subplot(1, 2, 1)

# Scatter plot with color based on density
scatter_train = plt.scatter(y_train, model.predict(X_train).flatten(), cmap='viridis', alpha=0.7)
plt.plot(y_train, y_train, 'r-', linewidth=3)  # Add a red diagonal line

# Add MSE and R2 as text annotations (formatted to 4 decimal places)
plt.text(0.05, 0.95, f'MSE: {mse_train:.4f}\nR2: {r2_train:.4f}', transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', backgroundcolor='white')

plt.title(f'Training Data Parody Plot (DNN)')
plt.xlabel('Actual Rg Values', fontsize=15)
plt.ylabel('Predicted Rg Values', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim([min(y_train), max(y_train)])
plt.ylim([min(y_train), max(y_train)])
cbar_train = plt.colorbar(scatter_train)
cbar_train.set_label('Data Point Density', fontsize=15)
cbar_train.ax.tick_params(labelsize=13)

# Parody plot for test data
plt.subplot(1, 2, 2)

# Scatter plot with color based on density
scatter_test = plt.scatter(y_test, model.predict(X_test).flatten(), cmap='viridis', alpha=0.7)
plt.plot(y_test, y_test, 'r-', linewidth=3)  # Add a red diagonal line

# Add MSE and R2 as text annotations (formatted to 4 decimal places)
plt.text(0.05, 0.95, f'MSE: {mse_test:.4f}\nR2: {r2_test:.4f}', transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', backgroundcolor='white')

plt.title(f'Test Data Parody Plot (DNN)')
plt.xlabel('Actual Rg Values', fontsize=15)
plt.ylabel('Predicted Rg Values', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim([min(y_test), max(y_test)])
plt.ylim([min(y_test), max(y_test)])
cbar_test = plt.colorbar(scatter_test)
cbar_test.set_label('Data Point Density', fontsize=15)
cbar_test.ax.tick_params(labelsize=13)

plt.tight_layout()
plt.show()
