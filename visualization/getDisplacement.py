import numpy as np
import plotly.graph_objects as go

# Function to parse CSV for X and Y values
def parse_csv_for_xy(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    x_vals = data[:, 0]  # Original X
    y_vals = data[:, 1]  # Original Y
    return x_vals, y_vals

# Function to get original points (optional, if needed)
def get_original_points():
    # Example: Replace with actual logic if needed
    return np.array([]), np.array([])

# Input and output file paths
input_file = '../output/gradient_estimate_notch_free_bottom_displacement.csv'

# Deformed points (Ux, Uy)
data = np.loadtxt(input_file, delimiter=',', skiprows=1)
x_vals = data[:, 0]  # Original X
y_vals = data[:, 1]  # Original Y
deformed_x = data[:, 2]  # Ux is already the deformed X position
deformed_y = data[:, 3]  # Uy is already the deformed Y position

# Create a figure for plotting
fig = go.Figure()

# Add scatter plot for original points
fig.add_trace(go.Scatter3d(
    x=x_vals, y=y_vals, z=np.zeros_like(x_vals), mode='markers',
    marker=dict(size=4, color='blue'), name='Original Points'
))

# Add scatter plot for deformed points
fig.add_trace(go.Scatter3d(
    x=deformed_x, y=deformed_y, z=np.zeros_like(deformed_x), mode='markers',
    marker=dict(size=4, color='red'), name='Deformed Points'
))

# Update layout for better visualization
fig.update_layout(scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z',
    xaxis=dict(range=[-0.3, 1.2]),
    zaxis=dict(range=[-0.01, 0.01]),
    aspectmode='cube'
))

# Show the plot and save to file
fig.show()
