import matplotlib.pyplot as plt

# Data
data_points = [200, 500, 1000, 2000, 3000, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 14366]

# R2 scores for different models
xgboost_r2 = [-0.1132, -0.0932, -0.0366, 0.1201, 0.1764, 0.7608, 0.7890, 0.8392, 0.8486, 0.8938, 0.9188, 0.9177, 0.9360]
random_forest_r2 = [-0.2299, -0.1945, -0.10, -0.0030, 0.065, 0.3240, 0.3937, 0.4779, 0.4803, 0.5668, 0.5750, 0.6346, 0.6425]
dnn_r2 = [0.5615, 0.5751, 0.5810, 0.6397, 0.6824, 0.8929, 0.9145, 0.9263, 0.9356, 0.9397, 0.9401, 0.9411, 0.9412]

# Desired y-axis tick locations
desired_y_ticks = [-0.2,0.2, 0.6, 1.0]

# Plotting
plt.figure(figsize=(6, 6))

# Plot lines with specific colors
plt.plot(data_points, xgboost_r2, marker='o', label='XGBoost', color='blue', linewidth=3, markersize=10, alpha=0.7)
plt.plot(data_points, random_forest_r2, marker='s', label='Random Forest', color='skyblue', linewidth=3, markersize=10, alpha=0.9)
plt.plot(data_points, dnn_r2, marker='d', label='DNN', color='purple', linewidth=3, markersize=10, alpha=0.5)

plt.xlabel('Number of Data Points', fontsize=20)
plt.ylabel('R2 Score Testing', fontsize=20)
plt.title('Number of Data Points vs. R2 Test Score', fontsize=18)

# Darken the rectangular box outline and axes
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2.5)

plt.legend(frameon=False)  # Remove legend box outline
plt.grid(False)

# Set the desired x-axis tick locations
plt.xticks([2000, 6000, 10000, 14000], fontsize=14)

# Set the desired y-axis tick locations
plt.yticks(desired_y_ticks, fontsize=14)

plt.show()
