import csv
import numpy as np
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
input_file = '../output/gradient_estimate_notch_free_bottom_deformation_gradient.csv'

def calculate_largest_eigenvalue(row):
    x, y, f11, f12, f21, f22 = map(float, (row['X'], row['Y'], row['F11'], row['F12'], row['F21'], row['F22']))
    matrix = np.array([[f11, f12], [f21, f22]])
    matrix -= np.eye(2)  
    strain = 1/2 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvals(strain)
    norm = max(abs(eigenvalues))
    return norm

def calculate_largest_norm(row):
    x, y, f11, f12, f21, f22 = map(float, (row['X'], row['Y'], row['F11'], row['F12'], row['F21'], row['F22']))
    matrix = np.array([[f11, f12], [f21, f22]])
    matrix -= np.eye(2)  # Subtract the identity matrix
    strain = 1/2 * (matrix + matrix.T)
    try:
        norm = np.linalg.norm(strain, ord=2)
        if (x> 0.9):
            print(f"Warning: Large norm detected at ({x}, {y}) with f11={f11}, f12={f12}, f21={f21}, f22={f22}. Norm: {norm}")
    except np.linalg.LinAlgError as e:
        norm = float('nan') 
    return norm

def calculate_largest_eigenvector(row):
    x, y, f11, f12, f21, f22 = map(float, (row['X'], row['Y'], row['F11'], row['F12'], row['F21'], row['F22']))
    matrix = np.array([[f11, f12], [f21, f22]])
    matrix -= np.eye(2)  
    strain = 1/2 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eig(strain)
    max_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    return eigenvectors[:, max_eigenvalue_index]
            
import matplotlib.pyplot as plt

x_values = []
y_values = []
gradient_norms = []

with open(input_file, mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        try:
            x = float(row['X'])
            y = float(row['Y'])
            gradient_norm = calculate_largest_norm(row)
            if np.isnan(gradient_norm):
                continue
            if not (np.isnan(x) or np.isnan(y)):
                x_values.append(x)
                y_values.append(y)
                gradient_norms.append(gradient_norm)
        except ValueError:
            continue
print(min(gradient_norms), max(gradient_norms))

# Create grid data for interpolation
xi = np.linspace(min(x_values), max(x_values), 100)
yi = np.linspace(min(y_values), max(y_values), 100)
x_min, x_max = np.min(x_values), np.max(x_values)
y_min, y_max = np.min(y_values), np.max(y_values)
grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_values = griddata((x_values, y_values), gradient_norms, (grid_x, grid_y), method='linear')
norm = Normalize(vmin=np.min(gradient_norms), vmax=np.max(gradient_norms))
plt.figure(figsize=(8, 6))
contour = plt.contourf(grid_x, grid_y, grid_values, levels=100, cmap='viridis', norm=norm)
plt.colorbar(contour, label='Norm')

# Scatter plot of the coordinates with their strain norms
plt.scatter(x_values, y_values, c=gradient_norms, cmap='viridis', edgecolors='k', s=100, marker='o', label='Coordinates', norm=norm)

# Set the range for the colorbar
# plt.clim(1, 1.2)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Norm Visualization')
plt.legend()
plt.show()