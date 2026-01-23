#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:53:57 2023

@author: tarak
"""

import numpy as np
import matplotlib.pyplot as plt

# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targetsafe.txt', delimiter=',')

# Calculate the number of bins and their widths
num_bins = 9  # Half of the original number of bins
bin_width = 1.3 # Adjust the width as needed to achieve the desired distances

# Generate a histogram
plt.figure(figsize=(8, 8))
hist, bins, _ = plt.hist(target_data, bins=num_bins, color='skyblue', edgecolor='skyblue', width=bin_width)

plt.xlabel('Adsorption Free Energy', fontsize=25)
plt.ylabel('Number of Data Points', fontsize=20)


# Specify the positions and labels for the x-axis
plt.xticks([10, 14, 18,22,26], [ '10', '14', '18','22','26'], fontsize=18)

# Specify the positions and labels for the y-axis
plt.yticks([1000, 2000, 3000, 4000,5000], ['1000', '2000', '3000', '4000','5000'], fontsize=18)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
plt.tight_layout()
plt.show()
