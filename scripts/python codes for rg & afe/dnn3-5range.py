import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam  # Import Adam optimizer
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
# Senp.random.seed(42)
np.random.seed(0)
tf.random.set_seed(0)

# Load predictor data from the new directory (modify the file path)
predictor_data = np.genfromtxt('/home/tarak/hashmi/predictors.txt', delimiter=',')
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')

# Create MinMaxScaler object with custom feature range
scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit and transform the data
data = [[1], [2]]
scaled_data = scaler.fit_transform(predictor_data)

consecutive_lengths_sequence = [[] for _ in range(len(scaled_data))]

# Maximum block possible
max_block = len(scaled_data[0])
max_block_size = max_block
Block_input = np.zeros((len(scaled_data),len(scaled_data[0])+1))

for j in range(len(scaled_data)):
    current = None
    count = 0
    for i in range(len(scaled_data[0])):
        #print(scaled_data[0,i])
        if scaled_data[j,i] == current:
            count += 1
            current = scaled_data[j,i]
        else:
            if current is not  None:
                # print(count,current)
                consecutive_lengths_sequence[j].append(count*current) 
            current = scaled_data[j,i]
            count = 1
    Block_input[j,0] = len(consecutive_lengths_sequence[j])
    for k in range(len(consecutive_lengths_sequence[j])):
        Block_input[j,k+1] = consecutive_lengths_sequence[j][k]

Block_input = Block_input/max_block
# predictors_train, predictors_test, targets_train, targets_test = train_test_split(Block_input, target_data, test_size=0.2)

# predictors_train_norm = predictors_train
# predictors_test_norm  = predictors_test
n_cols = Block_input.shape[1]       
#%%

# Filter data for training (values between 3 and 5)
train_indices = np.where((target_data >= 3.0) & (target_data <= 5))[0][:14366]
X_train = Block_input[train_indices]
y_train = target_data[train_indices]

# Filter data for testing (values outside the range 3 to 5)
test_indices = np.where((target_data < 3) | (target_data > 5))[0]
X_test = Block_input[test_indices]
y_test = target_data[test_indices]

# Create the DNN model with dropout and set learning rate to 0.01
dnn_model = Sequential([
    Dense(50, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(25, activation='relu'),
    
    Dense(1)
])

# Define the optimizer with the desired learning rate
optimizer = Adam(learning_rate=0.001)

# Compile the model with the specified optimizer
dnn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Train the model
history = dnn_model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0, verbose=1)

# Make predictions
y_pred_train = dnn_model.predict(X_train).flatten()
y_pred_test = dnn_model.predict(X_test).flatten()

# Calculate the R-squared (R2) score for both datasets
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Print the R2 scores
print(f'R2 Score (Training): {r2_train:.4f}')
print(f'R2 Score (Testing): {r2_test:.4f}')


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
    
plt.xlabel('Actual Free energy Values', fontsize=25)
plt.ylabel('Predicted Free Energy Values', fontsize=25)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])
plt.grid(False)
plt.xticks([3, 4, 5, 6], fontsize=18, weight='bold')
plt.yticks([3, 4, 5, 6], fontsize=18, weight='bold')

plt.tight_layout()
plt.show()

