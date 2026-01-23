#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:45:05 2023

@author: tarak
"""

import matplotlib.pyplot as plt

# Data
ranges = ['5', '10', '15', '20']
r2_scores_rf = [-0.1590, -0.0197, 0.1536, 0.2256]
r2_scores_xgb = [0.0146, 0.3234, 0.6427, 0.7317]
r2_scores_dnn = [0.3693, 0.7389, 0.8965, 0.9521]

# Desired y-axis tick locations
desired_y_ticks = [-0.2,0.2, 0.6, 1.0]

# Plotting
plt.figure(figsize=(6, 6))

# Plot lines with specific colors
plt.plot(ranges, r2_scores_xgb, marker='o', label='XGBoost', color='blue', linewidth=3, markersize=10, alpha=0.7)
plt.plot(ranges, r2_scores_rf, marker='s', label='Random Forest', color='red', linewidth=3, markersize=10, alpha=0.9)
plt.plot(ranges, r2_scores_dnn, marker='d', label='DNN', color='purple', linewidth=3, markersize=10, alpha=0.9)

plt.xlabel('% Range  ', fontsize=20,fontweight='bold')
plt.ylabel('R2 Score Testing', fontsize=20,fontweight='bold')

# Darken the rectangular box outline and axes
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(4)

plt.legend(frameon=False)  # Remove legend box outline
plt.grid(False)

# Set the desired x-axis tickhrre ibnhave t ad thiese stuff locations
plt.xticks([ 0,1,2,3], fontsize=14)

# Set the desired y-axis tick locations
plt.yticks(desired_y_ticks, fontsize=14)

plt.show()
