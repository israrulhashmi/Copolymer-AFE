import matplotlib.pyplot as plt

# Data
data_points = [100, 200, 300, 500, 1000, 2000, 3000, 5000, 7000, 8438]

# R2 test scores for each model (XGBoost, Random Forest, DNN treated as ANN)
r2_xgboost = [0.1447, 0.3803, 0.8328, 0.8782, 0.8921, 0.8990, 0.9131, 0.9287, 0.9379, 0.94]
r2_random_forest = [0.0214, 0.0646, 0.1044, 0.1474, 0.1910, 0.2243, 0.2450, 0.2825, 0.3194, 0.3352]
r2_dnn = [0.2523, 0.6136, 0.8334, 0.8857, 0.9509, 0.9727, 0.9788, 0.9788, 0.9851, 0.9833]  # Treated as DNN
# Desired y-axis tick locations
desired_y_ticks = [0,0.25, 0.50,0.75, 1.0]

# Plotting
plt.figure(figsize=(6, 6))

# Plot lines with specific colors
plt.plot(data_points, r2_xgboost, marker='o', label='XGBoost', color='blue', linewidth=3, markersize=10, alpha=0.9)
plt.plot(data_points, r2_random_forest, marker='s', label='Random Forest', color='red', linewidth=3, markersize=10, alpha=0.9)
plt.plot(data_points, r2_dnn, marker='d', label='DNN', color='purple', linewidth=3, markersize=10, alpha=0.9)

plt.xlabel('Number of Data Points', fontsize=20,fontweight='bold')
plt.ylabel('R2 Score Testing', fontsize=20,fontweight='bold')


# Darken the rectangular box outline and axes
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(4)

plt.legend(frameon=False)  # Remove legend box outline
plt.grid(False)

# Set the desired x-axis tick locations
plt.xticks([500,2000, 4000, 6000, 8000], fontsize=14)

# Set the desired y-axis tick locations
plt.yticks(desired_y_ticks, fontsize=14)

plt.show()