import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import io

# === Load displacement data ===
name = "gradient_estimate_notch_neumann_first_hit_boundary"
data = pd.read_csv(f'../output/{name}_displacement.csv', header=None).values
x = data[:, 0]
y = data[:, 1]
u = data[:, 2]
v = data[:, 3]

# Round to avoid floating point precision issues
rounded_x = np.round(x, 6)
rounded_y = np.round(y, 6)

# Group by y for horizontal gradients
horizontal_groups = defaultdict(list)
for xi, yi, ui, vi in zip(rounded_x, rounded_y, u, v):
    horizontal_groups[yi].append((xi, yi, ui, vi))

# Group by x for vertical gradients
vertical_groups = defaultdict(list)
for xi, yi, ui, vi in zip(rounded_x, rounded_y, u, v):
    vertical_groups[xi].append((xi, yi, ui, vi))

# Sort all groups
for key in horizontal_groups:
    horizontal_groups[key] = sorted(horizontal_groups[key], key=lambda tup: tup[0])  # sort by x
for key in vertical_groups:
    vertical_groups[key] = sorted(vertical_groups[key], key=lambda tup: tup[1])  # sort by y

# Collect gradients
gradient_x, gradient_y = [], []
dudx, dvdx, dudy, dvdy = [], [], [], []

# Compute horizontal (x) derivatives
for row in horizontal_groups.values():
    if len(row) < 3:
        continue
    for i in range(1, len(row) - 1):
        x_prev, y0, u_prev, v_prev = row[i - 1]
        x_curr, _, u_curr, v_curr = row[i]
        x_next, _, u_next, v_next = row[i + 1]
        dx = x_next - x_prev
        if dx == 0:
            continue
        gradient_x.append(x_curr)
        gradient_y.append(y0)
        dudx.append((u_next - u_prev) / dx)
        dvdx.append((v_next - v_prev) / dx)

# Compute vertical (y) derivatives and store in a map
dudy_map = {}
dvdy_map = {}
for col in vertical_groups.values():
    if len(col) < 3:
        continue
    for i in range(1, len(col) - 1):
        x0, y_prev, u_prev, v_prev = col[i - 1]
        _, y_curr, u_curr, v_curr = col[i]
        _, y_next, u_next, v_next = col[i + 1]
        dy = y_next - y_prev
        if dy == 0:
            continue
        key = (x0, y_curr)
        dudy_map[key] = (u_next - u_prev) / dy
        dvdy_map[key] = (v_next - v_prev) / dy

# Match vertical gradients to horizontal sample points
final_dudy = []
final_dvdy = []
for xi, yi in zip(gradient_x, gradient_y):
    key = (xi, yi)
    final_dudy.append(dudy_map.get(key, 0.0))  # fallback to 0 if missing
    final_dvdy.append(dvdy_map.get(key, 0.0))

# Convert to arrays
gradient_x = np.array(gradient_x)
gradient_y = np.array(gradient_y)
dudx = np.array(dudx)
dvdx = np.array(dvdx)
dudy = np.array(final_dudy)
dvdy = np.array(final_dvdy)

# === Load deformation gradient data ===
with open(f'../output/{name}_deformation_gradient.csv', 'r') as f:
    lines = [line for line in f if not line.startswith("X,Y")]
data2 = pd.read_csv(io.StringIO(''.join(lines)), header=None,
                    names=["X", "Y", "F11", "F12", "F21", "F22"])
data2.dropna(inplace=True)

x2 = data2["X"].values
y2 = data2["Y"].values
F11 = pd.to_numeric(data2["F11"].values, errors='coerce')
F12 = pd.to_numeric(data2["F12"].values, errors='coerce')
F21 = pd.to_numeric(data2["F21"].values, errors='coerce')
F22 = pd.to_numeric(data2["F22"].values, errors='coerce')

# === Create map from (x, y) to estimated gradients ===
rounded_coords = np.round(np.column_stack((gradient_x, gradient_y)), 6)
gradient_map = {
    (x_, y_): (dudx_, dudy_, dvdx_, dvdy_)
    for (x_, y_), dudx_, dudy_, dvdx_, dvdy_ in zip(
        rounded_coords, dudx, dudy, dvdx, dvdy
    )
}

# === Match and compute differences ===
diff_results = []
for xi, yi, f11, f12, f21, f22 in zip(x2, y2, F11, F12, F21, F22):
    xi_r, yi_r = round(xi, 6), round(yi, 6)
    dudx_val, dudy_val, dvdx_val, dvdy_val = gradient_map.get((xi_r, yi_r), (np.nan, np.nan, np.nan, np.nan))
    diff_results.append([
        xi, yi,
        dudx_val - f11 if not np.isnan(dudx_val) else np.nan,
        dudy_val - f12 if not np.isnan(dudy_val) else np.nan,
        dvdx_val - f21 if not np.isnan(dvdx_val) else np.nan,
        dvdy_val - f22 if not np.isnan(dvdy_val) else np.nan
    ])

# === Filter differences greater than 0.05 ===
filtered_diff_results = [
    row for row in diff_results
    if any(abs(val) > 0.1 for val in row[2:] if not np.isnan(val))
]

# === Save filtered differences ===
filtered_output_df = pd.DataFrame(filtered_diff_results, columns=['X', 'Y', 'dudx-F11', 'dudy-F12', 'dvdx-F21', 'dvdy-F22'])
filtered_output_df.dropna(inplace=True)
filtered_output_df.to_csv(f'../output/{name}_filtered_gradient_differences.csv', index=False)

