import csv
import numpy as np
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
input_file = '../output/gradient_estimate_free_boundary_gravity_deformation_gradient.csv'
output_file = '../output/gradient_estimate_free_boundary_gravity_deformation_gradient_norm.csv'

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
        print(f"Row {row}: norm = {norm}")
    except np.linalg.LinAlgError as e:
        # print(f"Error computing norm for row {row}: {e}")
        norm = float('nan')  # Assign NaN if SVD does not converge
    return norm

def calculate_largest_eigenvector(row):
    x, y, f11, f12, f21, f22 = map(float, (row['X'], row['Y'], row['F11'], row['F12'], row['F21'], row['F22']))
    matrix = np.array([[f11, f12], [f21, f22]])
    matrix -= np.eye(2)  
    strain = 1/2 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eig(strain)
    max_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    return eigenvectors[:, max_eigenvalue_index]

with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ['X', 'Y', 'F11', 'F12', 'F21', 'F22']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    next(reader) 
    for i, row in enumerate(reader):
        try:
            x = float(row['X'])
            y = float(row['Y'])
            if not (np.isnan(x) or np.isnan(y)):
                norm = calculate_largest_norm(row)
                writer.writerow({'X': row['X'], 'Y': row['Y'], 'norm': row['norm']})
        except ValueError:
            continue
            
import matplotlib.pyplot as plt

x_values = []
y_values = []
gradient_norms = []

with open(output_file, mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        try:
            x = float(row['X'])
            y = float(row['Y'])
            norm = float(row['norm'].strip('()').split('+')[0])
            if not (np.isnan(x) or np.isnan(y) or np.isnan(norm)):
                x_values.append(x)
                y_values.append(y)
                gradient_norms.append(norm)
        except ValueError:
            continue

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

# Labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Norm Visualization')
plt.legend()
plt.show()

# Normalize eigenvectors for visualization
# eigenvector_x = []
# eigenvector_y = []



# with open(output_file, mode='r') as infile:
#     reader = csv.DictReader(infile)
#     for row in reader:
#         try:
#             x = float(row['X'])
#             y = float(row['Y'])
#             eigenvector = calculate_largest_eigenvector(row)
#             if not (np.isnan(x) or np.isnan(y) or np.any(np.isnan(eigenvector))):
#                 x_values.append(x)
#                 y_values.append(y)
#                 eigenvector_x.append(eigenvector[0])
#                 eigenvector_y.append(eigenvector[1])
#         except ValueError:
#             continue

# # Normalize eigenvectors for consistent arrow lengths
# eigenvector_x = np.array(eigenvector_x)
# eigenvector_y = np.array(eigenvector_y)
# magnitude = np.sqrt(eigenvector_x**2 + eigenvector_y**2)
# eigenvector_x /= magnitude
# eigenvector_y /= magnitude

# # Plot eigenvectors as quiver plot
# plt.figure(figsize=(8, 6))
# plt.quiver(x_values, y_values, eigenvector_x, eigenvector_y, cmap='viridis', scale=20)
# plt.colorbar(label='eigenvalue')

# # Labels and title
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Eigenvector Visualization')
# plt.show()