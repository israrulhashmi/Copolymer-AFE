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

# Filter data for training (values between 3.5 and 4)
train_indices = np.where((target_data >= 14.0) & (target_data <= 18))
X_train = predictor_data[train_indices]
y_train = target_data[train_indices]

# Filter data for testing (values less than 3 and greater than 5)
test_indices = np.where((target_data < 14) | (target_data > 18))
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Check if there are samples in the test set before proceeding
if len(X_test) == 0:
    print("No samples found in the test set.")
else:
    # Create the XGBoost regression model with learning rate and max depth settings
    xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=4, random_state=10)

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

# Define the dark pink color using RGB values

# Create a scatter plot for in-range test data
# Create Fig and gridspec
fig = plt.figure(figsize=(16, 10), dpi= 80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

# Define the axes
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])



# Decorations
ax_main.set(title='Scatterplot with Histograms \n displ vs hwy', xlabel='displ', ylabel='hwy')
ax_main.title.set_fontsize(20)
for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
    item.set_fontsize(14)

xlabels = ax_main.get_xticks().tolist()
ax_main.set_xticklabels(xlabels)
plt.show()


