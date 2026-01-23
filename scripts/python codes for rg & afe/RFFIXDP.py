import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')

# Initialize a list to store R2 scores for Random Forest
r2_scores_randomforest = []

# Ranges for training
training_ranges = [(3, 5)]

# Number of data points for training in each range
num_points_per_range = 1000

# Set the specific hyperparameters
hyperparameters = {
    'n_estimators': 1000,
    'max_depth': 8,
    'min_samples_split': 4,
    'min_samples_leaf': 2,
    'random_state': 42
}

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

    # Create the Random Forest regression model with specific hyperparameters
    rf_model = RandomForestRegressor(**hyperparameters)

    # Fit the model
    rf_model.fit(X_train, y_train)

    # Predict using the model
    y_pred_randomforest = rf_model.predict(X_test)

    # Compute R2 score
    r2_randomforest = r2_score(y_test, y_pred_randomforest)
    r2_scores_randomforest.append(r2_randomforest)
    
    # Print R2 test scores for Random Forest
    print(f'R2 Score (Testing) for {min_range} to {max_range} range (Random Forest): {r2_randomforest:.4f}')

# Create a comparison plot of R2 scores vs. Training Range for Random Forest
plt.figure(figsize=(10, 6))

# Plot R2 scores for Random Forest
plt.plot([f'{min_range}-{max_range}' for min_range, max_range in training_ranges], r2_scores_randomforest, marker='x', label='Random Forest')

plt.title('r squared value score vs training range')
plt.xlabel('training range', fontsize=15)
plt.ylabel('R2 score', fontsize=15)
plt.xticks(rotation=45)
plt.legend()
plt.grid(False)

plt.tight_layout()
plt.show()
