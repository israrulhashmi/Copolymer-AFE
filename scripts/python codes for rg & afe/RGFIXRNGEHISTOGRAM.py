import numpy as np
import matplotlib.pyplot as plt

# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targets.txt', delimiter=',')

# Calculate the number of bins and their widths
num_bins = 15  # Half of the original number of bins
bin_width = 0.15  # Adjust the width as needed to achieve the desired distances

# Generate a histogram
plt.figure(figsize=(8, 8))
hist, bins, patches = plt.hist(target_data, bins=num_bins, color='red', edgecolor='red', width=bin_width)

# Identify the indices corresponding to the bins between 3 and 5
green_bins = np.where((bins >= 3) & (bins <= 5))

# Change the color of the identified bins to green
for idx in green_bins[0]:
    patches[idx].set_fc('green')
    patches[idx].set_edgecolor('green')  # Set edge color for green bars

plt.xlabel('Radius of Gyration', fontsize=25)
plt.ylabel('Number of Data Points', fontsize=25)

# Specify the positions and labels for the x-axis
plt.xticks([3, 4, 5, 6], ['3', '4', '5', '6'], fontsize=18,weight='bold')

# Specify the positions and labels for the y-axis
plt.yticks([500, 2000, 4000], ['500', '2000', '4000'], fontsize=18,weight='bold')

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(4)
plt.tight_layout()
plt.show()
