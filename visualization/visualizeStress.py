import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
# Lamé parameters
# lmbda = 100
# mu = 80
filename = "lame_wob_adjoint_50_displacement_gradient"

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

# Save stress components to CSV file
stress_csv_filename = f'../output/{filename}_stress.csv'
stress_df = pd.DataFrame({
    'X': x,
    'Y': y,
    'sigma_xx': sigma_xx,
    'sigma_yy': sigma_yy,
    'sigma_xy': sigma_xy,
    'mu': mu,
    'nu': nu,
    'lambda': lmbda
})
stress_df.to_csv(stress_csv_filename, index=False)
print(f"Stress values saved to {stress_csv_filename}")

# Save stress components in 2x2 matrix format to text file
output_filename = f'../output/{filename}_stress_matrices.txt'
with open(output_filename, 'w') as f:
    for i in range(len(sigma_xx)): 
        f.write(f"\nStress matrix at point ({x[i]}, {y[i]}):\n")
        f.write(f"[{sigma_xx[i]:.6f}  {sigma_xy[i]:.6f}]\n")
        f.write(f"[{sigma_xy[i]:.6f}  {sigma_yy[i]:.6f}]\n")
        f.write("-" * 30 + "\n")


# Compute stress matrix for each point
max_eigenvalues = []
stress_00 = []
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
    stress_00.append(sigma_xx[i])

max_eigenvalues = np.array(max_eigenvalues)
stress_00 = np.array(stress_00)

# Visualization with interpolated contour plot
plt.figure(figsize=(12, 8))

# Create triangulation for interpolation
tri = Triangulation(x, y)

# Create interpolator for getting stress values at any point
interpolator = LinearTriInterpolator(tri, stress_00)

# Create interpolated contour plot
contour = plt.tricontourf(tri, stress_00, levels=50, cmap='viridis', extend='both', vmax=1.5)
plt.colorbar(contour, label='Stress σ_xx (Pa)')

# Add contour lines for better visualization
contour_lines = plt.tricontour(tri, stress_00, levels=20, colors='black', alpha=0.3, linewidths=0.5)
plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')

plt.title('Interpolated Stress σ_xx Distribution (Click to see stress value)')
plt.xlabel('x')
plt.ylabel('y')

# Text annotation for displaying clicked values
text_annotation = plt.text(0.02, 0.98, '', transform=plt.gca().transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=10)

# Click event handler
def on_click(event):
    if event.inaxes is None:
        return
    
    if event.button == 1:  # Left mouse button
        click_x = event.xdata
        click_y = event.ydata
        
        # Interpolate stress value at clicked point
        stress_value = interpolator(click_x, click_y)
        
        if not np.isnan(stress_value):
            # Update text annotation
            text_annotation.set_text(f'Position: ({click_x:.4f}, {click_y:.4f})\nStress σ_xx: {stress_value:.6f} Pa')
            print(f'Clicked at ({click_x:.4f}, {click_y:.4f}): Stress σ_xx = {stress_value:.6f} Pa')
            plt.draw()
        else:
            text_annotation.set_text(f'Position: ({click_x:.4f}, {click_y:.4f})\nStress: Outside domain')
            print(f'Clicked at ({click_x:.4f}, {click_y:.4f}): Outside domain')
            plt.draw()

# Connect the click event
plt.gcf().canvas.mpl_connect('button_press_event', on_click)

plt.tight_layout()
plt.autoscale(enable=True, axis='both', tight=True,)
plt.show()

