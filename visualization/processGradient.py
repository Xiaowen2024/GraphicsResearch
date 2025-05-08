import numpy as np

# Load original gradients (keep all rows)
original_gradients = np.genfromtxt(
    '../output/gradient_estimate_notch_neumann_first_hit_boundary_deformation_gradient.csv',
    delimiter=',',
    skip_header=1,
    invalid_raise=False
)

# Load gradient differences
gradient_differences = np.genfromtxt(
    '../output/gradient_estimate_notch_neumann_first_hit_boundary_filtered_gradient_differences.csv',
    delimiter=',',
    skip_header=1
)

# Build a fast lookup from (x, y) -> diff
diff_map = {
    (round(diff[0], 6), round(diff[1], 6)): diff[2:]
    for diff in gradient_differences
    if not np.isnan(diff[0]) and not np.isnan(diff[1])
}

# Restore gradients
restored_gradients = []
for grad in original_gradients:
    if len(grad) != 6:
        restored_gradients.append([np.nan]*6)
        continue

    x, y, dudx, dudy, dvdx, dvdy = grad
    key = (round(x, 6), round(y, 6))
    if key in diff_map:
        dudx_diff, dudy_diff, dvdx_diff, dvdy_diff = diff_map[key]
        dudx = dudx + dudx_diff if not np.isnan(dudx) else np.nan
        dudy = dudy + dudy_diff if not np.isnan(dudy) else np.nan
        dvdx = dvdx + dvdx_diff if not np.isnan(dvdx) else np.nan
        dvdy = dvdy + dvdy_diff if not np.isnan(dvdy) else np.nan
    restored_gradients.append([x, y, dudx, dudy, dvdx, dvdy])

# Save with header, preserving nan and original row count
np.savetxt(
    '../output/gradient_estimate_notch_neumann_first_hit_boundary_restored_gradients.csv',
    restored_gradients,
    delimiter=',',
    header='X,Y,F11,F12,F21,F22',
    comments='',
    fmt='%.6f'
)
