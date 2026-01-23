import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictorsafe.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targetsafe.txt', delimiter=',')

# Filter data for testing (values between 11 to 13 and 16 to 18)
test_indices = np.where(((target_data >= 12) & (target_data <= 13)) | ((target_data >= 13) & (target_data <= 14)))
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Define different training data point counts
training_point_counts = [100, 200, 300, 500, 1000, 2000, 3000, 5000, 7000, 8438,15000,20000]

# Initialize a list to store R2 for Linear Regression
r2_linear_regression_list = []

# Iterate over different training point counts
for num_points in training_point_counts:
    # Filter data for training (values between 13 and 17)
    train_indices = np.where((target_data >= 13) & (target_data <= 14))[0][:num_points]
    X_train = predictor_data[train_indices]
    y_train = target_data[train_indices]

    # Create and train the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    r2_linear_regression_list.append(r2_lr)

    # Print R2 for the current data subset
    print(f'R2 for {num_points} data points: {r2_lr:.4f}')

# Plot the R2 for testing data for Linear Regression
plt.figure(figsize=(10, 6))
plt.plot(training_point_counts, r2_linear_regression_list, label='Linear Regression', marker='o', color='green')
plt.xlabel('Training Data Point Count')
plt.ylabel('R2 (Testing)')
plt.title('R2 (Testing) for Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
