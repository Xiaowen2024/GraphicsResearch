import matplotlib.pyplot as plt

# Hardcoded crack points
left_crack = [
    (0.753737, -0.018967),
    (1.05026, -0.0051341),
    (-0.0504569, -0.0067439),
    (0.243944, -0.0238515),
    (0.753737, -0.018967)
]

crack_tip = [
    (0.781033, 0.0462629),
    (1.09487, 0.0497357),
    (-0.0932561, 0.0495431),
    (0.223851, 0.0439444),
    (0.781033, 0.0462629)
]

right_crack = [
    (0.846263, 0.018967),
    (1.14974, 0.0051341),
    (-0.149543, 0.0067439),
    (0.156056, 0.0238515),
    (0.846263, 0.018967)
]

# Dirichlet boundary points
boundary_dirichlet = [
    { "x": 0.8, "y": 0 },
    { "x": 1.1, "y": 0 },
    { "x": 1, "y": 1 },
    { "x": 0, "y": 1 },
    { "x": -0.1, "y": 0 },
    { "x": 0.2, "y": 0 },
    { "x": 0.5, "y": 0.2 },
    { "x": 0.8, "y": 0 }
]

# Convert to X, Y lists
left_x, left_y = zip(*left_crack)
tip_x, tip_y = zip(*crack_tip)
right_x, right_y = zip(*right_crack)

boundary_x = [pt["x"] for pt in boundary_dirichlet]
boundary_y = [pt["y"] for pt in boundary_dirichlet]

# Plotting
plt.figure(figsize=(8, 8))

# Plot boundary polygon
plt.plot(boundary_x, boundary_y, 'c-', linewidth=2, label='Boundary Dirichlet')

# Plot crack points
plt.scatter(left_x, left_y, color='red', label='Left Crack', s=40)
plt.scatter(tip_x, tip_y, color='green', label='Crack Tip', s=40)
plt.scatter(right_x, right_y, color='blue', label='Right Crack', s=40)

# Set axes
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.gca().set_aspect('equal', adjustable='box')

# Labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Crack Points over Dirichlet Boundary')
plt.grid(True)
plt.legend()
plt.show()
