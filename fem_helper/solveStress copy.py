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
with io.XDMFFile(MPI.COMM_WORLD, "notched_plate_mesh_2.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

with io.XDMFFile(MPI.COMM_WORLD, "notched_plate_facet_region_2.xdmf", "r") as xdmf:
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

# Dirichlet BC on top boundary (tag 2)
# dirichlet_id = 2
# u_D = Function(V)
# u_D.interpolate(lambda x: np.stack([0.1 * (x[0] - 0.5), np.zeros_like(x[1])]) if dim == 2 else
#                             np.stack([0.1 * (x[0] - 0.5), np.zeros_like(x[1]), np.zeros_like(x[2])]))
# facets = np.where(facet_tags.values == dirichlet_id)[0]
# dofs = locate_dofs_topological(V, domain.topology.dim - 1, facet_tags.indices[facets])
# bc = dirichletbc(u_D, dofs)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), epsilon(v)) * dx
L = ufl.dot(Constant(domain, PETSc.ScalarType((0.0,) * dim)), v) * ufl.ds

a_form = fem.form(a)
L_form = fem.form(L)

u_bc_left = fem.Function(V)
u_bc_left.interpolate(lambda x: np.stack((-0.1 * np.ones_like(x[0]), np.zeros_like(x[1]))))

u_bc_right = fem.Function(V)
u_bc_right.interpolate(lambda x: np.stack((0.1 * np.ones_like(x[0]), np.zeros_like(x[1]))))

def on_left(x): return np.isclose(x[0], 0.0)
def on_right(x): return np.isclose(x[0], 1.0)

dofs_left = fem.locate_dofs_geometrical(V, on_left)
dofs_right = fem.locate_dofs_geometrical(V, on_right)

bc_left = fem.dirichletbc(u_bc_left, dofs_left)
bc_right = fem.dirichletbc(u_bc_right, dofs_right)

def on_top(x): return np.isclose(x[1], 1.0)
dofs_top = fem.locate_dofs_geometrical(V.sub(1), on_top)  # sub(1) → y component

zero = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
zero.interpolate(lambda x: np.zeros_like(x[0]))
bc_top = fem.dirichletbc(zero, dofs_top, V.sub(1))  # only constrain u_y

bcs = [bc_left, bc_right, bc_top]

# Assemble system
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

b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs)

# Solve
uh = fem.Function(V)
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.CG)
ksp.getPC().setType(PETSc.PC.Type.HYPRE)
ksp.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()
# uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# el1 = element("Lagrange", domain.ufl_cell().cellname(), 1, shape=(2,))
# V1 = fem.functionspace(domain, el1)
# uh_out = fem.Function(V1)
# uh_out.interpolate(uh)

# # Save mesh and interpolated function
# with io.XDMFFile(domain.comm, "displacement.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(uh_out)

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_values = uh.x.array.real.reshape((u_grid.n_points, -1))

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

# Attach displacement components to the grid
u_grid.point_data["u_x"] = u_values[:, 0]
u_grid.point_data["u_y"] = u_values[:, 1]

# Compute strain and stress tensors
strain_values = compute_strain(u_values, u_geometry, dim)
stress_values = compute_stress(strain_values, lmbda, mu, dim)

# Attach stress components to the grid
u_grid.point_data["sigma_xx"] = stress_values[:, 0, 0]
u_grid.point_data["sigma_yy"] = stress_values[:, 1, 1]

# Compute stress tensor at each point
# elementStress = element("DG", domain.ufl_cell().cellname(), 0, shape=(dim, dim))
# stress_space = fem.functionspace(domain, elementStress)
# stress_evaluated = fem.Function(stress_space)
# stress_expr = fem.Expression(sigma(uh), stress_space.element.interpolation_points())
# stress_evaluated.interpolate(stress_expr)

# # Extract stress components for visualization
# stress_values = stress_evaluated.x.array.reshape((-1, dim, dim))
# print("Stress values shape:", stress_values.shape)
# sigma_xx = stress_values[:, 0, 0]
# sigma_yy = stress_values[:, 1, 1]

# # Attach stress components to the grid
# # Interpolate stress tensor to a point-based function space
# elementStress = element("CG", domain.ufl_cell().cellname(), 1, shape=(dim, dim))
# point_stress_space = fem.functionspace(domain, elementStress)
# point_stress = fem.Function(point_stress_space)
# point_stress_expr = fem.Expression(sigma(uh), point_stress_space.element.interpolation_points())
# point_stress.interpolate(point_stress_expr)

# # Extract stress components for visualization
# point_stress_values = point_stress.x.array.reshape((-1, dim, dim))
# sigma_xx = point_stress_values[:, 0, 0]
# sigma_yy = point_stress_values[:, 1, 1]

# # Attach stress components to the grid
# u_grid.point_data["sigma_xx"] = sigma_xx
# u_grid.point_data["sigma_yy"] = sigma_yy


# Plot u_x
plotter_ux = pyvista.Plotter(title="u_x")
plotter_ux.add_mesh(u_grid.copy(), scalars="u_x", show_edges=True)
plotter_ux.view_xy()
plotter_ux.show()

# Plot u_y
plotter_uy = pyvista.Plotter(title="u_y")
plotter_uy.add_mesh(u_grid.copy(), scalars="u_y", show_edges=True)
plotter_uy.view_xy()
plotter_uy.show()

# Plot σ_xx
plotter_sigma_xx = pyvista.Plotter(title="Stress σ_xx")
plotter_sigma_xx.add_mesh(u_grid.copy(), scalars="sigma_xx", show_edges=True)
plotter_sigma_xx.view_xy()
plotter_sigma_xx.show()

# Plot σ_yy
plotter_sigma_yy = pyvista.Plotter(title="Stress σ_yy")
plotter_sigma_yy.add_mesh(u_grid.copy(), scalars="sigma_yy", show_edges=True)
plotter_sigma_yy.view_xy()
plotter_sigma_yy.show()

# --- Compute and visualize principal stress σ₁ (max) ---

# Define stress again (if needed)
# stressEl = element("DG", domain.ufl_cell().cellname(), 0, shape=(dim, dim))
# W = fem.functionspace(domain, el)
# stress = fem.Function(W)
# S = ufl.TrialFunction(W)
# T = ufl.TestFunction(W)
# print("dim =", domain.geometry.dim)     
# print("cell =", domain.ufl_cell())

# assert S.ufl_shape == (dim, dim), f"Expected (dim, dim), got {S.ufl_shape}"

# # 3. Stress tensor (sigma(uh)) projection
# a_proj = ufl.inner(S, T) * dx
# L_proj = ufl.inner(sigma(uh), T) * dx

# # 4. Assemble system
# A_proj = assemble_matrix(form(a_proj))
# A_proj.assemble()
# b_proj = assemble_vector(form(L_proj))

# # 5. Solve linear system
# solver = PETSc.KSP().create(domain.comm)
# solver.setOperators(A_proj)
# solver.setType("cg")
# solver.getPC().setType("jacobi")

# solver.solve(b_proj, stress.x.petsc_vec)
# stress.x.scatter_forward() 
# # Project the stress tensor expression into the space
# # stress = fem.project(sigma(uh), W)

# # Compute per-cell principal stress (σ₁)
# stress_values = stress.x.array.reshape((-1, dim * dim))
# sigma_max_vals = np.zeros(len(stress_values))

# for i, S in enumerate(stress_values):
#     S_mat = np.array([
#         [S[0], 0.5 * (S[1] + S[2])],
#         [0.5 * (S[1] + S[2]), S[3]]
#     ])
#     eigvals = np.linalg.eigvalsh(S_mat)
#     sigma_max_vals[i] = eigvals[1]  # max principal stress

# # Attach to cells
# tdim = domain.topology.dim
# domain.topology.create_connectivity(tdim, tdim - 1)
# topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# grid.cell_data["sigma_max"] = sigma_max_vals

# # Plot σ₁
# plotter_sigma = pyvista.Plotter(title="Max Principal Stress σ₁")
# plotter_sigma.add_mesh(grid.copy(), scalars="sigma_max", show_edges=True)
# plotter_sigma.view_xy()
# plotter_sigma.show()

# el0 = element("DG", domain.ufl_cell().cellname(), 0, shape=(dim,dim))
# W0 = fem.functionspace(domain, el0)
# expr = fem.Expression(sigma(uh), W0.element.interpolation_points())
# stress_evaluated = fem.Function(W0)
# stress_evaluated.interpolate(expr)

# # 9. Extract σ_xx and σ_yy for visualization
# stress_values = stress_evaluated.x.array.reshape((-1, 2, 2))
# sigma_xx = stress_values[:, 0, 0]
# sigma_yy = stress_values[:, 1, 1]

# # 10. Plot stress using PyVista
# topology, cell_types, geometry = plot.vtk_mesh(W0)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# grid.cell_data["sigma_xx"] = sigma_xx
# grid.cell_data["sigma_yy"] = sigma_yy

# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, scalars="sigma_xx", show_edges=True)
# plotter.view_xy()
# plotter.show()

elementStress = element("DG", domain.ufl_cell().cellname(), 0, shape=(dim, dim))
W = fem.functionspace(domain, elementStress)
stress = fem.Function(W)

expr = fem.Expression(sigma(u), W.element.interpolation_points())
stress.interpolate(expr)
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, 0)
topology, cell_types, geometry = vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, domain.geometry.x)

# Attach stress components
stress_array = stress.x.array.reshape((-1, dim * dim))
for i in range(dim):
    for j in range(dim):
        component = stress_array[:, i * dim + j]
        grid.point_data[f"stress_{i}{j}"] = component

# Plot
plotter = pyvista.Plotter()
plotter.add_mesh(grid, scalars="stress_00", show_edges=True)
plotter.show()