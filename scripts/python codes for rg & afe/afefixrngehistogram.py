import numpy as np
import matplotlib.pyplot as plt

# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targetsafe.txt', delimiter=',')

# Calculate the number of bins and their widths
num_bins = 11  # Half of the original number of bins
bin_width = 0.99  # Adjust the width as needed to achieve the desired distances

# Generate a histogram
plt.figure(figsize=(8, 8))
hist, bins, patches = plt.hist(target_data, bins=num_bins, color='red', edgecolor='red', width=bin_width)

# Identify the indices corresponding to the bins between 3 and 5
green_bins = np.where((bins >= 13.8) & (bins <= 18))

# Change the color of the identified bins to green
for idx in green_bins[0]:
    patches[idx].set_fc('green')
    patches[idx].set_edgecolor('green')  # Set edge color for green bars

plt.xlabel('Adsorption Free Energy', fontweight='bold',fontsize=25)
plt.ylabel('Number of Data Points', fontweight='bold',fontsize=25)

# Specify the positions and labels for the x-axis
plt.xticks([10, 14, 18, 22, 25], ['10', '14', '18', '22', '25'], fontsize=18,weight='bold')

# Specify the positions and labels for the y-axis
plt.yticks([500, 2000, 4000], ['500', '2000', '4000'], fontsize=18,weight='bold')
red_patch = plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor='red')
green_patch = plt.Rectangle((0, 0), 1, 1, fc="green", edgecolor='green')

# Add legend
plt.legend([red_patch, green_patch], ['Test Data', 'Train Data'], fontsize=18,loc='upper right',frameon=False)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(4)
plt.tight_layout()