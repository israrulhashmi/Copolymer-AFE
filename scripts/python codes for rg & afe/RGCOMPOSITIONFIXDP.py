#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/test a simple DNN on the 80/20 (8020) copolymer dataset.

Assumptions:
- predictors file: 8020.txt (rows = polymer chains, cols = monomer sequence encoded as 0/1)
- property file  : 8020_property.txt (one target value per row, aligned with 8020.txt)
- You want to randomly pick exactly 4000 datapoints, then 80/20 split, no other conditions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -------------------- Reproducibility --------------------
np.random.seed(42)
tf.random.set_seed(42)

# -------------------- Paths (edit if needed) --------------------
base_dir = "/home/tarak/hashmi/rgdata/rgcopolymers/compositions"
seq_path  = os.path.join(base_dir, "5050.txt")
prop_path = os.path.join(base_dir, "5050y.txt")

# -------------------- Load data --------------------
X_all = np.genfromtxt(seq_path, delimiter=",", dtype=float)
y_all = np.genfromtxt(prop_path, delimiter=",", dtype=float)

# Ensure proper shapes
if X_all.ndim == 1:
    X_all = X_all.reshape(1, -1)
y_all = np.asarray(y_all).reshape(-1)

if X_all.shape[0] != y_all.shape[0]:
    raise ValueError(
        f"Row mismatch: predictors have {X_all.shape[0]} rows but property has {y_all.shape[0]} rows."
    )

N = X_all.shape[0]
if N < 1000:
    raise ValueError(f"8020 dataset has only {N} rows; cannot sample 4000 without replacement.")

# -------------------- Select exactly 4000 points --------------------
idx = np.random.choice(N, size=1000, replace=False)
X = X_all[idx]
y = y_all[idx]

# -------------------- 80/20 train-test split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"Loaded 8020 data: total_rows={N}, selected={X.shape[0]}")
print(f"Train size: {X_train.shape[0]}")
print(f"Test size : {X_test.shape[0]}")

# -------------------- Build DNN model --------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

# -------------------- Train --------------------
epochs = 1000
history = model.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=1)

# -------------------- Predict --------------------
y_pred_train = model.predict(X_train).reshape(-1)
y_pred_test  = model.predict(X_test).reshape(-1)

# -------------------- Metrics --------------------
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train  = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test  = r2_score(y_test, y_pred_test)

rmse_test = np.sqrt(mse_test)

print(f"R2 Score (Training): {r2_train:.4f}")
print(f"R2 Score (Testing) : {r2_test:.4f}")
print(f"RMSE on Test Set   : {rmse_test:.4f}")

# -------------------- Parity plot (match your earlier styling) --------------------
plt.figure(figsize=(8.05, 8.13))

plt.scatter(y_test, y_pred_test, c='none', marker='v', edgecolors='red',
            alpha=0.9, label='Test Data', s=200)
plt.scatter(y_train, y_pred_train, c='none', marker='o', edgecolors='green',
            alpha=1.0, label='Train Data', s=160)

min_val = min(np.min(y_train), np.min(y_test))
max_val = max(np.max(y_train), np.max(y_test))
plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=4)

plt.axis('equal')

ax = plt.gca()
for side in ['top', 'right', 'bottom', 'left']:
    ax.spines[side].set_linewidth(4)
    ax.spines[side].set_color('black')

legend = plt.legend(frameon=False)
for label in legend.get_texts():
    label.set_fontweight('bold')

plt.xlabel('Actual Free energy Values', fontsize=25)
plt.ylabel('Predicted Free Energy Values', fontsize=25)

plt.xticks([3, 4, 5, 6], fontsize=18, weight='bold')
plt.yticks([3, 4, 5, 6], fontsize=18, weight='bold')

plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])
plt.grid(False)

plt.tight_layout()
plt.show()
