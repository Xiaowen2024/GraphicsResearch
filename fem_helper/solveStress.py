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

# Load mesh and boundary tags
with io.XDMFFile(MPI.COMM_WORLD, "notched_plate_mesh.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

with io.XDMFFile(MPI.COMM_WORLD, "notched_plate_facet_region.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(domain, name="name_to_read")

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
dirichlet_id = 2
u_D = Function(V)
u_D.interpolate(lambda x: np.stack([0.1 * (x[0] - 0.5), np.zeros_like(x[1])]) if dim == 2 else
                            np.stack([0.1 * (x[0] - 0.5), np.zeros_like(x[1]), np.zeros_like(x[2])]))
facets = np.where(facet_tags.values == dirichlet_id)[0]
dofs = locate_dofs_topological(V, domain.topology.dim - 1, facet_tags.indices[facets])
bc = dirichletbc(u_D, dofs)

# Variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), epsilon(v)) * dx
L = ufl.dot(Constant(domain, PETSc.ScalarType((0.0,) * dim)), v) * ufl.ds

a_form = fem.form(a)
L_form = fem.form(L)

# Assemble system
A = fem.petsc.assemble_matrix(a_form, bcs=[bc])
A.assemble()
b = fem.petsc.assemble_vector(L_form)

# 🛡️ Safe lifting with full signature and memory checks
b_array = np.asarray(b.array, dtype=np.float64, order="C")
apply_lifting(
    b.array,             # RHS as NumPy array
    [form(a)],           # list of forms
    [[bc]]
)

b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Solve
uh = fem.Function(V)
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.CG)
ksp.getPC().setType(PETSc.PC.Type.HYPRE)
ksp.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()
# uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


el1 = element("Lagrange", domain.ufl_cell().cellname(), 1, shape=(2,))
V1 = fem.functionspace(domain, el1)
uh_out = fem.Function(V1)
uh_out.interpolate(uh)

# # Save mesh and interpolated function
# with io.XDMFFile(domain.comm, "displacement.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(uh_out)

tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")
    
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real.reshape((u_grid.n_points, -1))
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    u_figure = u_plotter.screenshot("displacement_field.png")
