file_path = '../output/gradient_estimate_notch_neumann_first_hit_boundary_calculated_gradients.csv'  # Replace with your actual file path

header_line = 'X,Y,F11,F12,F21,F22\n'

# Read all lines first
with open(file_path, 'r') as f:
    lines = f.readlines()

# Modify every other line starting from the first
for i in range(0, len(lines), 2):
    lines[i] = header_line

# Write back to the same file
with open(file_path, 'w') as f:
    f.writelines(lines)
