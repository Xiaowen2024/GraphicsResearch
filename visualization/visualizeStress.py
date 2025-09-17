import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Lamé parameters
# lmbda = 100
# mu = 80
filename = "lame_wob_forward_zero_displacement_deformation_gradient"

# Load CSV file
data = pd.read_csv(f'../output/{filename}.csv')

# Extract deformation gradient components
x = np.array(data['X'])
y = np.array(data['Y'])

dudx = data['F11'].apply(pd.to_numeric, errors='coerce').to_numpy()
dudy = data['F12'].apply(pd.to_numeric, errors='coerce').to_numpy()
dvdx = data['F21'].apply(pd.to_numeric, errors='coerce').to_numpy()
dvdy = data['F22'].apply(pd.to_numeric, errors='coerce').to_numpy()

# Compute strain components
epsilon_xx = dudx 
epsilon_yy = dvdy
epsilon_xy = 1/2 * (dudy + dvdx)

E = 1.0 
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# Compute stress components
sigma_xx = lmbda * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_xx
sigma_yy = lmbda * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_yy
sigma_xy = 2 * mu * epsilon_xy

assert len(sigma_xx) == len(sigma_yy) == len(sigma_xy) == len(x) == len(y)

# # Print stress components in 2x2 matrix format
for i in range(len(sigma_xx)): 
    print(f"\nStress matrix at point ({x[i]}, {y[i]}):")
    print(f"[{sigma_xx[i]:.2f}  {sigma_xy[i]:.2f}]")
    print(f"[{sigma_xy[i]:.2f}  {sigma_yy[i]:.2f}]")
    print("-" * 30)

# Compute stress matrix for each point
max_eigenvalues = []
for i in range(len(sigma_xx)):
    # Create 2x2 stress matrix
    # stress_matrix = np.array([[max(0, sigma_xx[i]), max(0, sigma_xy[i])],
                            # [max(0, sigma_xy[i]), max(0, sigma_yy[i])]])
    stress_matrix = np.array([[sigma_xx[i], sigma_xy[i]],[sigma_xy[i], sigma_yy[i]]])
    
    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eig(stress_matrix)
    eigenvalues_clipped = np.clip(eigenvalues, a_min=0, a_max=None)
    
    
    # Get maximum eigenvalue (principal stress)
    max_eigenvalues.append(np.max(eigenvalues_clipped))

max_eigenvalues = np.array(max_eigenvalues)
# Visualization
plt.figure(figsize=(12, 8))

# Maximum Principal Stress (Highest Eigenvalue, original values)
plt.scatter(x, y, c=max_eigenvalues, cmap='viridis', vmin=0.1, vmax=1.5)
plt.colorbar(label='Maximum Principal Stress (Pa)')
plt.title('Maximum Principal Stress Distribution (Bounded Plot)')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.autoscale(enable=True, axis='both', tight=True)
plt.show()
