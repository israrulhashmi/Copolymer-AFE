import numpy as np
import matplotlib.pyplot as plt

# Load target data from the new directory (modify the file path)
target_data = np.genfromtxt('/home/tarak/hashmi/targetsafe.txt', delimiter=',')

# Calculate the number of bins and their widths
num_bins = 11  # Half of the original number of bins
bin_width = 0.99  # Adjust the width as needed to achieve the desired distances

# Generate a histogram
plt.figure(figsize=(8, 8))
hist, bins, _ = plt.hist(target_data, bins=num_bins, color='skyblue', edgecolor='skyblue', width=bin_width)

plt.xlabel('Radius of Gyration', fontsize=25)
plt.ylabel('Number of Data Points', fontsize=20)


# Specify the positions and labels for the x-axis
plt.xticks([10, 14, 18, 22,25], ['10', '14', '18', '22','25'], fontsize=18)

# Specify the positions and labels for the y-axis
plt.yticks([0, 2000,  4000], [ '0','2000',  '4000'], fontsize=18)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
plt.tight_layout()
plt.show()
