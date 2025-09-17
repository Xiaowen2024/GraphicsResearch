import numpy as np
import ufl
from dolfinx import fem, mesh, io, plot
from dolfinx.fem import (Function, dirichletbc, locate_dofs_topological, Constant,
                         form, apply_lifting, assemble_matrix, assemble_vector, set_bc)
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from ufl import dx
from dolfinx.fem.petsc import assemble_vector
import pyvista
from dolfinx.plot import vtk_mesh

# Load mesh and boundary tags
with io.XDMFFile(MPI.COMM_WORLD, "notched_plate_mixed.xdmf", "r") as xdmf:
     domain = xdmf.read_mesh(name="Grid")
     domain.topology.create_entities(1)

with io.XDMFFile(MPI.COMM_WORLD, "notched_plate_mixed_facet_region.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(domain, name="Grid")


# Define vector function space
dim = domain.geometry.dim
el = element("Lagrange", domain.ufl_cell().cellname(), 2, shape=(dim,))
V = fem.functionspace(domain, el)

# Material constants
E = 1.0
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# Strain and stress
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(dim) + 2 * mu * epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), epsilon(v)) * dx

a_form = fem.form(a)

def on_left(x):
    return np.isclose(x[0], 0.0)

def on_right(x):
    return np.isclose(x[0], 1.0)

# Left: displacement (-1, 0)
u_left = fem.Function(V)
u_left.interpolate(lambda x: np.stack((0 * np.ones_like(x[0]), np.zeros_like(x[1]))))

# Right: displacement (1, 0)
u_right = fem.Function(V)
u_right.interpolate(lambda x: np.stack((0 * np.ones_like(x[0]), np.zeros_like(x[1]))))

# Locate DOFs
left_dofs = fem.locate_dofs_geometrical(V, on_left)
right_dofs = fem.locate_dofs_geometrical(V, on_right)

print(f"# Left DOFs: {len(left_dofs)}, Right DOFs: {len(right_dofs)}")

# Apply Dirichlet BCs
bc_left = dirichletbc(u_left, left_dofs)
bc_right = dirichletbc(u_right, right_dofs)

bcs = [bc_left, bc_right]

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# # Traction vectors
# traction_top = fem.Constant(domain, PETSc.ScalarType((0.0, -1.0)))
# traction_bottom = fem.Constant(domain, PETSc.ScalarType((0.0, 1.0)))

# # Add both Neumann contributions
# L = ufl.dot(traction_top, v) * ds(3) + ufl.dot(traction_bottom, v) * ds(1)
# L_form = fem.form(L)

n = ufl.FacetNormal(domain)

# Define a scalar pressure value (e.g., p=1.0). 
# This will be applied inward on both surfaces.
pressure = fem.Constant(domain, PETSc.ScalarType(0))

# Define the linear form L (the right-hand side of the weak form)
# The traction vector is t = -p * n
# The contribution to the weak form is integral(dot(t, v)*ds)

# Contribution from the top boundary (marked as 3)
L_top = ufl.dot(-pressure * n, v) * ds(3)

# Contribution from the bottom boundary (marked as 1)
L_bottom = ufl.dot(-pressure * n, v) * ds(1)

# Total linear form
L = L_top + L_bottom
L_form = fem.form(L)

A = fem.petsc.assemble_matrix(a_form, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector(L_form)

# 🛡️ Safe lifting with full signature and memory checks
b_array = np.asarray(b.array, dtype=np.float64, order="C")
apply_lifting(
    b.array,             # RHS as NumPy array
    [form(a)],           # list of forms
    [bcs]
)

b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, bcs)

# Solve
uh = fem.Function(V)
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.CG)
ksp.getPC().setType(PETSc.PC.Type.HYPRE)
ksp.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_values = uh.x.array.real.reshape((u_grid.n_points, -1))

top_mask = np.abs(u_geometry[:, 1] - 1.0) < 1e-6
bottom_mask = np.abs(u_geometry[:, 1]) < 1e-6

# Compute strain tensor directly from displacement values
def compute_strain(u_values, points, dim):
    num_points = points.shape[0]
    strain = np.zeros((num_points, dim, dim))

    # Assume uniform spacing in the domain
    for i in range(1, num_points - 1):
        grad_u = np.zeros((dim, dim))
        for j in range(dim):
            # Central difference approximation of ∂u_j / ∂x_i
            for k in range(dim):
                dx = points[i + 1, k] - points[i - 1, k]
                if dx != 0:
                    grad_u[k, j] = (u_values[i + 1, j] - u_values[i - 1, j]) / dx
        strain[i] = 0.5 * (grad_u + grad_u.T)

    return strain

# Compute stress tensor directly from strain
def compute_stress(strain, lmbda, mu, dim):
    stress = np.zeros_like(strain)
    for i in range(strain.shape[0]):
        trace_strain = np.trace(strain[i])
        stress[i] = lmbda * trace_strain * np.eye(dim) + 2 * mu * strain[i]
    return stress

u_values = uh.x.array.reshape((u_grid.n_points, -1))
u_grid.point_data["u_x"] = u_values[:, 0]
u_grid.point_data["u_y"] = u_values[:, 1]
# Attach displacement components to the grid
u_grid.point_data["u_x"] = u_values[:, 0]
u_grid.point_data["u_y"] = u_values[:, 1]

# Compute strain and stress tensors
strain_values = compute_strain(u_values, u_geometry, dim)
stress_values = compute_stress(strain_values, lmbda, mu, dim)

# Attach stress components to the grid
u_grid.point_data["sigma_xx"] = stress_values[:, 0, 0]
u_grid.point_data["sigma_yy"] = stress_values[:, 1, 1]

mask = u_geometry[:, 1] > -1  # or 0.01, depending on your notch depth

# Extract filtered data
filtered_grid = u_grid.extract_points(mask, adjacent_cells=True)

# Plot u_x
plotter_ux = pyvista.Plotter(title="u_x")
plotter_ux.add_mesh(filtered_grid.copy(), scalars="u_x", show_edges=True)
plotter_ux.view_xy()
plotter_ux.show()

# Plot u_y
plotter_uy = pyvista.Plotter(title="u_y")
plotter_uy.add_mesh(filtered_grid.copy(), scalars="u_y", show_edges=True)
plotter_uy.view_xy()
plotter_uy.show()

elementStress = element("DG", domain.ufl_cell().cellname(), 0, shape=(dim, dim))
W = fem.functionspace(domain, elementStress)
stress = fem.Function(W)

points = W.element.interpolation_points()
expr = fem.Expression(sigma(uh), points)
stress.interpolate(expr)
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, 0)
topology, cell_types, geometry = vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, domain.geometry.x)

# Attach stress components
# Interpolate stress to a point-based function space
elementStressPoint = element("CG", domain.ufl_cell().cellname(), 1, shape=(dim, dim))
W_point = fem.functionspace(domain, elementStressPoint)
stress_point = fem.Function(W_point)

# Interpolate stress values
points = W_point.element.interpolation_points()
expr_point = fem.Expression(sigma(uh), points)
stress_point.interpolate(expr_point)

# Attach stress components to the grid
stress_point_array = stress_point.x.array.reshape((-1, dim, dim))
for i in range(dim):
    for j in range(dim):
        grid.point_data[f"stress_{i}{j}"] = stress_point_array[:, i, j]

# Plot
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, scalars="stress_00", show_edges=True)
# plotter.view_xy()  # Set the view to face the straight side
# plotter.show()

stress_fields = {}
for i in range(dim):
    for j in range(dim):
        el_scalar = element("Lagrange", domain.ufl_cell().cellname(), degree=1)
        W_scalar = fem.functionspace(domain, el_scalar)
        stress_comp = fem.Function(W_scalar)
        expr = fem.Expression(ufl.as_tensor(sigma(uh)[i, j]), W_scalar.element.interpolation_points())
        stress_comp.interpolate(expr)
        field_name = f"stress_{i}{j}"
        stress_fields[field_name] = stress_comp.x.array
        grid.point_data[field_name] = stress_fields[field_name]

# Plot all in a single 2×2 layout (adjust if dim=3)
plotter = pyvista.Plotter(shape=(2, 2), title="Stress Components")
component_keys = list(stress_fields.keys())

# for idx, name in enumerate(component_keys):
#     plotter.subplot(idx // 2, idx % 2)
#     plotter.add_text(name, font_size=12)
#     plotter.add_mesh(grid.copy(), scalars=name, show_edges=True)
#     plotter.view_xy()

# plotter.link_views()  # Optional: synchronize camera
# plotter.show()

elementStressPoint = element("Lagrange", domain.ufl_cell().cellname(), 1, shape=(2, 2))
W_point = fem.functionspace(domain, elementStressPoint)
stress_tensor = fem.Function(W_point)
expr_point = fem.Expression(sigma(uh), W_point.element.interpolation_points())
stress_tensor.interpolate(expr_point)

# Step 2: Compute principal stresses (eigenvalues)
stress_array = stress_tensor.x.array.reshape((-1, 2, 2))  # Shape: (npoints, 2, 2)
principal_stresses = np.linalg.eigvalsh(stress_array)     # Shape: (npoints, 2), sorted ascending

# Step 3: Filter positive principal stresses
positive_principal_1 = np.maximum(principal_stresses[:, 1], 0)  # Max principal stress
positive_principal_2 = np.maximum(principal_stresses[:, 0], 0)  # Min principal stress (can be negative)

# Step 4: Add to PyVista grid and plot
grid.point_data["principal_max"] = positive_principal_1
grid.point_data["principal_min"] = positive_principal_2

# Visualization
plotter = pyvista.Plotter(shape=(1, 2), title="Principal Stresses (Positive Only)")

# plotter.subplot(0, 0)
plotter.add_text("Max Principal", font_size=12)
plotter.add_mesh(grid.copy(), scalars="principal_max", show_edges=True)
plotter.view_xy()

# plotter.subplot(0, 1)
# plotter.add_text("Min Principal (Clipped)", font_size=12)
# plotter.add_mesh(grid.copy(), scalars="principal_min", show_edges=True)
# plotter.view_xy()

# plotter.link_views()
plotter.show()
# Create plotter for displaced shape
# 1. Choose a scaling factor for visualization (10 is common for exaggeration)
scale = 1.0

# 2. Compute the new displaced geometry (only x and y)
displaced_geometry_2d = u_geometry[:, :2] + scale * u_values  # shape: (n_points, 2)

# 3. Pad with z=0 to make it (n_points, 3) for PyVista compatibility
displaced_geometry_3d = np.zeros_like(u_geometry)             # shape: (n_points, 3)
displaced_geometry_3d[:, :2] = displaced_geometry_2d

# 4. Assign displaced geometry to the grid
u_grid.points = displaced_geometry_3d

# 5. Optional: add magnitude of displacement as scalar
displacement_magnitude = np.linalg.norm(u_values, axis=1)
u_grid.point_data["|u|"] = displacement_magnitude

# 6. Plot
plotter = pyvista.Plotter(title="Deformed Mesh")
plotter.add_text(f"Deformed (scale={scale})", font_size=12)
plotter.add_mesh(u_grid.copy(), scalars="|u|", show_edges=True, cmap="plasma")
plotter.view_xy()
plotter.show()
