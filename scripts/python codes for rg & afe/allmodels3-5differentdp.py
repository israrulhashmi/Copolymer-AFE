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
test_indices = np.where((target_data < 3) & (target_data > 5))
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Initialize lists to store R2 scores
r2_scores_xgboost = []
r2_scores_randomforest = []
r2_scores_dnn = []
data_point_counts = []

# Loop through different training data point counts
for num_points in [200, 500, 1000, 2000, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 14366]:
    # Filter data for training (values between 3.5 and 4)
    train_indices = np.where((target_data >= 3.0) & (target_data <= 5))[0][:num_points]
    X_train = predictor_data[train_indices]
    y_train = target_data[train_indices]

    # Create the XGBoost regression model
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4,random_state=10)
    xgb_model.fit(X_train, y_train)
    y_pred_xgboost = xgb_model.predict(X_test)
    r2_xgboost = r2_score(y_test, y_pred_xgboost)
    r2_scores_xgboost.append(r2_xgboost)

    # Create the Random Forest regression model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=10)
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

    data_point_counts.append(num_points)

    # Print R2 for the current data subset
    print(f'R2 for {num_points} data points - XGBoost: {r2_xgboost:.4f}, Random Forest: {r2_randomforest:.4f}, DNN: {r2_dnn:.4f}')
