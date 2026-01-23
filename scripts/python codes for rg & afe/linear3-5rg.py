import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')

# Filter data for testing (values less than 3 and greater than 5)
test_indices = np.where((target_data < 3) | (target_data > 5))
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Initialize lists to store R2 scores and data point counts
r2_scores_linear = []
r2_scores_training = []  # Store training R2 scores
data_point_counts = []

# Loop through different training data point counts
for num_points in range(200, 14307):  # Start from 200, end at 14306, step by 100
    # Filter data for training (values between 3 and 5)
    train_indices = np.where((target_data >= 3) & (target_data <= 5))[0][:num_points]
    X_train = predictor_data[train_indices]
    y_train = target_data[train_indices]

    # Create the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Make predictions for training and test sets
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)

    # Calculate the R-squared (R2) score for training and test sets
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # Store R2 scores and data point count
    r2_scores_linear.append(r2_test)  # Append the R2 score for testing to the list
    r2_scores_training.append(r2_train)  # Append the R2 score for training to the list
    data_point_counts.append(num_points)

# Create a comparison plot of R2 scores vs. Data Point Counts
plt.figure(figsize=(12, 6))

# Plot R2 scores for Linear Regression
plt.plot(data_point_counts, r2_scores_linear, marker='o', linestyle='-', label='Testing R2')
plt.plot(data_point_counts, r2_scores_training, linestyle='--', label='Training R2')  # Add training R2 as a line

plt.title('R2 Score vs. Training Data Point Count for Extrapolation')
plt.xlabel('Training Data Point Count', fontsize=15)
plt.ylabel('R2 Score', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(np.arange(-0.3, 1.1, 0.1), fontsize=12)
plt.ylim([-0.3, 1])
plt.legend(fontsize=12)
plt.grid(False)

plt.tight_layout()
plt.show()
