import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')

# Filter data for training (values between 3 and 5)
train_indices = np.where((target_data >= 3.0) & (target_data <= 4))[0][:5000]
X_train = predictor_data[train_indices]
y_train = target_data[train_indices]

# Filter data for testing (values outside the range 3 to 5)
test_indices = np.where((target_data < 3) | (target_data > 4))[0]
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [500],  # Number of trees in the forest
    'max_depth': [100],  # Maximum depth of the trees
    'min_samples_split': [5,6],  # Minimum samples required to split a node
    'min_samples_leaf': [2, 4, 6]  # Minimum samples required at each leaf node
}

# Create the Random Forest regression model
rf_model = RandomForestRegressor(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2', n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found by the grid search
print("Best Parameters:")
print(grid_search.best_params_)

# Use the best model from the grid search
best_rf_model = grid_search.best_estimator_

# Train the best model
best_rf_model.fit(X_train, y_train)

# Make predictions using the best model
y_pred_train_best = best_rf_model.predict(X_train)
y_pred_test_best = best_rf_model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R2) score for both datasets
mse_train_best = mean_squared_error(y_train, y_pred_train_best)
r2_train_best = r2_score(y_train, y_pred_train_best)

mse_test_best = mean_squared_error(y_test, y_pred_test_best)
r2_test_best = r2_score(y_test, y_pred_test_best)

# Print R2 scores for the best model
print("\nR2 Score (Training) - Best Model:", r2_train_best)
print("R2 Score (Testing) - Best Model:", r2_test_best)

# Calculate data point density for coloring
xy_train = np.vstack([y_train, y_pred_train_best])
z_train = gaussian_kde(xy_train)(xy_train)

xy_test = np.vstack([y_test, y_pred_test_best])
z_test = gaussian_kde(xy_test)(xy_test)

# Rest of the code for plotting and visualization remains the same...
