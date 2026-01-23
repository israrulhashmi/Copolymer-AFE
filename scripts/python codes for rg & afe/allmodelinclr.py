import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')

# Filter data for training (values between 3 and 5)
train_indices = np.where((target_data >= 3) & (target_data <= 5))
X_train = predictor_data[train_indices]
y_train = target_data[train_indices]

# Filter data for testing (values less than 3 and greater than 5)
test_indices = np.where((target_data < 3) | (target_data > 5))
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Initialize lists to store R2 scores
r2_scores_xgboost = []
r2_scores_randomforest = []
r2_scores_dnn = []
r2_scores_linear = []  # Added for Linear Regression
data_point_counts = []

# Loop through different training data point counts
for num_points in [200, 500, 1000, 2000, 3000, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 14366]:
    # Filter data for training (values between 3 and 5)
    train_indices = np.where((target_data >= 3) & (target_data <= 5))[0][:num_points]

    # Check if the training dataset is empty for this training set size
    if len(train_indices) == 0:
        print(f"Skipping training for {num_points} training data points as the dataset is empty.")
        continue

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
        tf.keras.layers.Dense(32, activation='relu'),      # Fully connected hidden layer with ReLU activation
        tf.keras.layers.Dense(1)                           # Output layer (1 neuron for regression)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    y_pred_dnn = model.predict(X_test)
    r2_dnn = r2_score(y_test, y_pred_dnn)
    r2_scores_dnn.append(r2_dnn)

    # Create the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_linear = lr_model.predict(X_test)
    r2_linear = r2_score(y_test, y_pred_linear)
    r2_scores_linear.append(r2_linear)

    data_point_counts.append(num_points)

# Create a comparison plot of R2 scores vs. Data Point Counts
plt.figure(figsize=(12, 6))

# Plot R2 scores for XGBoost
plt.plot(data_point_counts, r2_scores_xgboost, marker='o', linestyle='-', label='XGBoost R2')

# Plot R2 scores for Random Forest
plt.plot(data_point_counts, r2_scores_randomforest, marker='x', linestyle='-', label='Random Forest R2')

# Plot R2 scores for DNN
plt.plot(data_point_counts, r2_scores_dnn, marker='s', linestyle='-', label='DNN R2')

# Plot R2 scores for Linear Regression
plt.plot(data_point_counts, r2_scores_linear, marker='d', linestyle='-', label='Linear Regression R2')

plt.title('R2 Score vs. Training Data Point Count for extrapolation')
plt.xlabel('Training Data Point Count', fontsize=15)
plt.ylabel('R2 Score', fontsize=15)
plt.xticks(data_point_counts)


# Create a comparison plot of R2 scores vs. Data Point Counts
plt.figure(figsize=(12, 6))

# Plot R2 scores for DNN (Training and Testing)
plt.plot(data_point_counts, r2_scores_dnn, marker='s', linestyle='-', label='DNN Training R2')
plt.annotate('DNN Training R2', xy=(data_point_counts[-1], r2_scores_dnn[-1]), xytext=(-50, 20), 
             textcoords='offset points', arrowprops=dict(arrowstyle="->"))

# Plot R2 scores for Linear Regression (Training and Testing)
plt.plot(data_point_counts, r2_scores_linear, marker='d', linestyle='-', label='Linear Regression R2')
plt.annotate('Linear Regression Training R2', xy=(data_point_counts[-1], r2_scores_linear[-1]), xytext=(-50, 20), 
             textcoords='offset points', arrowprops=dict(arrowstyle="->"))

plt.title('R2 Score vs. Training Data Point Count for DNN and Linear Regression')
plt.xlabel('Training Data Point Count', fontsize=15)
plt.ylabel('R2 Score', fontsize=15)
plt.xticks(data_point_counts)

# Add a legend
plt.legend()

plt.tight_layout()
plt.show()
# Plot R2 scores for XGBoost
plt.plot(data_point_counts, r2_scores_xgboost, marker='o', linestyle='-', label='XGBoost R2')

# Plot R2 scores for Random Forest
plt.plot(data_point_counts, r2_scores_randomforest, marker='x', linestyle='-', label='Random Forest R2')

# Plot R2 scores for DNN
plt.plot(data_point_counts, r2_scores_dnn, marker='s', linestyle='-', label='DNN R2')

# Plot R2 scores for Linear Regression
plt.plot(data_point_counts, r2_scores_linear, marker='d', linestyle='-', label='Linear Regression R2')

# Add model names next to markers
for i, num_points in enumerate(data_point_counts):
    plt.text(num_points, r2_scores_xgboost[i], 'XGBoost', ha='right', va='bottom')
    plt.text(num_points, r2_scores_randomforest[i], 'Random Forest', ha='right', va='bottom')
    plt.text(num_points, r2_scores_dnn[i], 'DNN', ha='right', va='bottom')
    plt.text(num_points, r2_scores_linear[i], 'Linear Regression', ha='right', va='bottom')

# Other plot settings remain unchanged
plt.title('R2 Score vs. Training Data Point Count for extrapolation')
plt.xlabel('Training Data Point Count', fontsize=15)
plt.ylabel('R2 Score', fontsize=15)
plt.xticks(data_point_counts)
plt.legend()
plt.tight_layout()
plt.show()

