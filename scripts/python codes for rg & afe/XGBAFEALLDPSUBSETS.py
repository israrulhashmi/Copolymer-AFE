import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictorsafe.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targetsafe.txt', delimiter=',')

# Filter data for testing (values less than 14 or greater than 18)
test_indices = np.where((target_data < 14) | (target_data > 18))
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Define different training data point counts
training_point_counts = [100, 200, 300, 500, 1000, 2000, 3000, 5000, 7000, 8438]

# Initialize lists to store R2 for each model
r2_xgboost_list = []
r2_random_forest_list = []
r2_linear_regression_list = []

# Iterate over different training point counts
for num_points in training_point_counts:
    # Filter data for training (values between 14 and 18)
    train_indices = np.where((target_data >= 14) & (target_data <= 18))[0][:num_points]
    X_train = predictor_data[train_indices]
    y_train = target_data[train_indices]

    # Create and train the XGBoost regression model
    xgb_model = XGBRegressor(n_estimators=300)
    xgb_model.fit(X_train, y_train)
    y_pred_xgboost = xgb_model.predict(X_test)
    r2_xgboost = r2_score(y_test, y_pred_xgboost)
    r2_xgboost_list.append(r2_xgboost)

    # Create and train the Random Forest regression model
    rf_model = RandomForestRegressor(n_estimators=300)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    r2_random_forest_list.append(r2_rf)

    # Create and train the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    r2_linear_regression_list.append(r2_lr)

# Plot the R2 for testing data for all models
plt.figure(figsize=(10, 6))
plt.plot(training_point_counts, r2_xgboost_list, label='XGBoost', marker='o')
plt.plot(training_point_counts, r2_random_forest_list, label='Random Forest', marker='o')
plt.plot(training_point_counts, r2_linear_regression_list, label='Linear Regression', marker='o')
plt.xlabel('Training Data Point Count')
plt.ylabel('R2 (Testing)')
plt.title('Comparison of R2 (Testing) for Different Models')
plt.legend()
plt.grid(True)
plt.show()
