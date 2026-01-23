import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')

# Filter data for training (values between 3 and 5)
train_indices = np.where((target_data >= 3.0) & (target_data <= 5))[0][:14000]
X_train = predictor_data[train_indices]
y_train = target_data[train_indices]

# Filter data for testing (values outside the range 3 to 5)
test_indices = np.where((target_data < 3) | (target_data > 5))[0]
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Create the XGBoost regression model with learning rate and max depth settings
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=3, random_state=10)  # You can tune hyperparameters here

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R2) score for both datasets
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print the R2 scores
print(f'R2 Score (Training): {r2_train:.4f}')
print(f'R2 Score (Testing): {r2_test:.4f}')

# Print the MSE for training and testing
print(f'Mean Squared Error (Training): {mse_train:.4f}')
print(f'Mean Squared Error (Testing): {mse_test:.4f}')

# Calculate data point density for coloring
xy_train = np.vstack([y_train, y_pred_train])
z_train = gaussian_kde(xy_train)(xy_train)

xy_test = np.vstack([y_test, y_pred_test])
z_test = gaussian_kde(xy_test)(xy_test)

# Total data points
total_data_points = len(predictor_data)
data_points_used_for_training = len(X_train)
data_points_used_for_testing = len(X_test)

print(f'Total Data Points: {total_data_points}')
print(f'Number of Data Points Used for Training: {data_points_used_for_training}')
print(f'Number of Data Points Used for Testing: {data_points_used_for_testing}')

# Create parody plots for both training and test datasets
plt.figure(figsize=(12, 5))

# Parody plot for training data
plt.subplot(1, 2, 1)

# Scatter plot with color based on density
scatter_train = plt.scatter(y_train, y_pred_train, c=z_train, cmap='viridis', alpha=0.7)
plt.plot(y_train, y_train, 'r-', linewidth=3)  # Add a red diagonal line

# Add MSE and R2 as text annotations (formatted to 4 decimal places)
plt.text(0.05, 0.95, f'MSE: {mse_train:.4f}\nR2: {r2_train:.4f}', transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', backgroundcolor='white')

plt.title(f'Training Data Parody Plot')
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
scatter_test = plt.scatter(y_test, y_pred_test, c=z_test, cmap='viridis', alpha=0.7)
plt.plot(y_test, y_test, 'r-', linewidth=3)  # Add a red diagonal line

# Add MSE and R2 as text annotations (formatted to 4 decimal places)
plt.text(0.05, 0.95, f'MSE: {mse_test:.4f}\nR2: {r2_test:.4f}', transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', backgroundcolor='white')

plt.title(f'Test Data Parody Plot')
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
