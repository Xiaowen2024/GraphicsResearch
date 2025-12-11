import numpy as np
import plotly.graph_objects as go

# Input and output file paths
input_file = '../output/lame_wob_adjoint_46_displacements.csv'

# Deformed points (Ux, Uy)
data = np.loadtxt(input_file, delimiter=',', skiprows=0)
x_vals = data[:, 0]  # Original X
y_vals = data[:, 1]  # Original Y
deformed_x = data[:, 2] + x_vals  # Ux is already the deformed X position
deformed_y = data[:, 3] + y_vals  # Uy is already the deformed Y position

# Create figures for plotting
fig1 = go.Figure()
fig2 = go.Figure()

# First figure: Original, deformed and displacement lines
# Add scatter plot for original points
fig1.add_trace(go.Scatter3d(
    x=x_vals, y=y_vals, z=np.zeros_like(x_vals), mode='markers',
    marker=dict(size=4, color='blue'), name='Original Points'
))

# Add scatter plot for deformed points
fig1.add_trace(go.Scatter3d(
    x=deformed_x, y=deformed_y, z=np.zeros_like(deformed_x), mode='markers',
    marker=dict(size=4, color='red'), name='Deformed Points'
))

# Add lines connecting original points to deformed points
for x_orig, y_orig, x_def, y_def in zip(x_vals, y_vals, deformed_x, deformed_y):
    fig1.add_trace(go.Scatter3d(
        x=[x_orig, x_def], y=[y_orig, y_def], z=[0, 0], mode='lines',
        line=dict(color='green', width=2), showlegend=False
    ))

# Second figure: Only deformed points
fig2.add_trace(go.Scatter3d(
    x=deformed_x, y=deformed_y, z=np.zeros_like(deformed_x), mode='markers',
    marker=dict(size=4, color='red'), name='Deformed Points'
))

# Update layout for both figures
for fig in [fig1, fig2]:
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        xaxis=dict(range=[-0.2, 1.2]),
        zaxis=dict(range=[-0.1, 1.1]),
        aspectmode='cube'
    ))

fig1.show()
fig2.show()
