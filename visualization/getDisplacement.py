import numpy as np
import plotly.graph_objects as go


# Input and output file paths
input_file = '../output/gradient_estimate_notch_neumann_first_hit_boundary_adative_sample_displacement.csv'

# Deformed points (Ux, Uy)
data = np.loadtxt(input_file, delimiter=',', skiprows=0)
x_vals = data[:, 0]  # Original X
y_vals = data[:, 1]  # Original Y
deformed_x = data[:, 2] + x_vals # Ux is already the deformed X position, add 1 to each value
deformed_y = data[:, 3] + y_vals  # Uy is already the deformed Y position, add 1 to each value

# Create a figure for plotting
fig = go.Figure()

# # Add scatter plot for original points
# fig.add_trace(go.Scatter3d(
#     x=x_vals, y=y_vals, z=np.zeros_like(x_vals), mode='markers',
#     marker=dict(size=4, color='blue'), name='Original Points'
# ))

# Check if deformed_x and deformed_y contain the value 0.0666667
# print("Deformed X values:", x_vals)
# print("Deformed Y values:", y_vals)
# Add scatter plot for deformed points
fig.add_trace(go.Scatter3d(
    x=deformed_x, y=deformed_y, z=np.zeros_like(deformed_x), mode='markers',
    marker=dict(size=2, color='red'), name='Deformed Points'
))


# Update layout for better visualization
fig.update_layout(scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z',
    xaxis=dict(range=[-0.3, 1.2]),
    zaxis=dict(range=[-0.1, 1.0]),
    aspectmode='cube'
))

fig.show()
