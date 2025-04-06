import csv
import numpy as np
from scipy.interpolate import griddata
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

input_file = '../output/gradient_estimate_notch_free_bottom_deformation_gradient.csv'

def calculate_eigenvalues_and_eigenvectors(row):
    x, y, f11, f12, f21, f22 = map(float, (row['X'], row['Y'], row['F11'], row['F12'], row['F21'], row['F22']))
    if np.isnan(f11) or np.isnan(f12) or np.isnan(f21) or np.isnan(f22):
        return None, None
    matrix = np.array([[f11, f12], [f21, f22]])
    matrix -= np.eye(2)  
    strain = 1/2 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eig(strain)
    return eigenvalues, eigenvectors

x_values = []
y_values = []
ellipses = []

with open(input_file, mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        try:
            x = float(row['X'])
            y = float(row['Y'])
            eigenvalues, eigenvectors = calculate_eigenvalues_and_eigenvectors(row)
            if eigenvalues is None or eigenvectors is None:
                continue
            if not (np.isnan(x) or np.isnan(y)):
                x_values.append(x)
                y_values.append(y)
                ellipses.append((eigenvalues, eigenvectors))
        except ValueError:
            continue
        
plt.figure(figsize=(8, 6))
ax = plt.gca()

# Normalize eigenvalues to ensure ellipses are appropriately scaled
scaling_factor = 0.1 # Adjust this factor to control the size of ellipses relative to the coordinates

for (x, y), (eigenvalues, eigenvectors) in zip(zip(x_values, y_values), ellipses):
    # Scale eigenvalues for visualization
    width = 2 * abs(eigenvalues[0]) * scaling_factor
    height = 2 * abs(eigenvalues[1]) * scaling_factor
    
    # Eigenvectors determine the orientation of the ellipse
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Create and add the ellipse
    ellipse = Ellipse((x, y), width, height, angle=angle)
    ax.add_patch(ellipse)
    ellipse.set_edgecolor('black')
    ellipse.set_facecolor('none')
    ellipse.set_linewidth(0.8)

# Scatter plot of the points
plt.scatter(x_values, y_values, color='red', s=10, label='Points')

# Labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ellipsoid Visualization (Scaled Eigenvalues and Eigenvectors)')
plt.legend()
plt.axis('equal')
plt.show()
