import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde


# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')
np.random.seed(0)
tf.random.set_seed(0)
# Create an array of indices within the range (3 to 5)
valid_indices = np.where((target_data >= 3.0) & (target_data <= 5))[0]

# Randomly select  data points within the range for training
selected_indices = np.random.choice(valid_indices,200, replace=False)
X_train = predictor_data[selected_indices]
y_train = target_data[selected_indices]

# Filter data for testing (values outside the range 3 to 5)
test_indices = np.where((target_data < 3) | (target_data > 5))[0]
X_test = predictor_data[test_indices]
y_test = target_data[test_indices]

# Create a simple DNN model using TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),      # Fully connected hidden layer with ReLU activation
    

   # Fully connected hidden layer with ReLU activation
    tf.keras.layers.Dense(1)                           # Output layer (1 neuron for regression)
])

# Compile the model with a learning rate of 0.01

optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])


# Train the DNN model for 100 epochs (you can change the number of epochs as needed)
epochs = 200
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
# Calculate the Mean Squared Error (MSE) and R-squared (R2) score for both datasets
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print the R2 scores
print(f'R2 Score (Training): {r2_train:.4f}')
print(f'R2 Score (Testing): {r2_test:.4f}')

# Calculate data point density for coloring
xy_train = np.vstack([y_train, y_pred_train.flatten()])
z_train = gaussian_kde(xy_train)(xy_train)

xy_test = np.vstack([y_test, y_pred_test.flatten()])
z_test = gaussian_kde(xy_test)(xy_test)

# Total data points
total_data_points = len(predictor_data)
data_points_used_for_training = len(X_train)
data_points_used_for_testing = len(X_test)

print(f'Total Data Points: {total_data_points}')
print(f'Number of Data Points Used for Training: {data_points_used_for_training}')
print(f'Number of Data Points Used for Testing: {data_points_used_for_testing}')

# Create parody plots for both training and test datasets
plt.figure(figsize=(8.05, 8.13))

# Scatter plot for test data with black color, darkened markers and larger marker size
plt.scatter(y_test, y_pred_test, c='none', marker='v', edgecolors='red', alpha=0.9, label='Test Data', s=200)

# Scatter plot for training data with green color, darkened markers and larger marker size
plt.scatter(y_train, y_pred_train, c='none', marker='o', edgecolors='green', alpha=1, label='Train Data', s=160)

# Add a red diagonal line from one corner to the opposite corner
min_val = min(min(y_train), min(y_test))
max_val = max(max(y_train), max(y_test))
plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=4)

# Set the aspect ratio to 'equal' for a full square box
plt.axis('equal')

# Darken the outer square lines by setting spines properties
ax = plt.gca()
ax.spines['top'].set_linewidth(4)
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth(4)
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_linewidth(4)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(4)
ax.spines['left'].set_color('black')


# Remove the outline of the legend box

legend = plt.legend(frameon=False)
for label in legend.get_texts():
    label.set_fontweight('bold')
    label.set_fontsize(18)
plt.xlabel('Actual Rg Values', fontsize=25)
plt.ylabel('Predicted Rg Values', fontsize=25)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])
plt.grid(False)
plt.xticks([3, 4, 5, 6], fontsize=18, weight='bold')
plt.yticks([3, 4, 5, 6], fontsize=18, weight='bold')

plt.tight_layout()
plt.show()
