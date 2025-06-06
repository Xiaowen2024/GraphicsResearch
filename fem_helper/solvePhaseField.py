import numpy as np
import ufl
from dolfinx import mesh, fem, plot, io
from mpi4py import MPI
from petsc4py import PETSc
import pyvista as pv
from dolfinx.plot import vtk_mesh
from basix.ufl import element
from ufl import grad, inner, dx, dot, sym, Identity, derivative

# --- Parameters ---
length = 1.0
height = 1.0
notch_width = 0.02
notch_height = 0.2
Nx, Ny = 80, 80  # Mesh resolution
Gc = 1.0  # Fracture energy
ell = 0.02  # Regularization length
E = 10.0
nu = 0.3
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# --- Mesh with bottom notch ---
xmin, xmax, ymin, ymax = 0.0, length, 0.0, height
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([xmin, ymin]), np.array([xmax, ymax])],
    [Nx, Ny],
    cell_type=mesh.CellType.triangle)

dim = domain.geometry.dim
el = element("Lagrange", domain.ufl_cell().cellname(), 1)
V_u = fem.functionspace(domain, el)
V_d = fem.functionspace(domain, el)

u = fem.Function(V_u, name="u")
d = fem.Function(V_d, name="d")

v = ufl.TestFunction(V_u)
w = ufl.TestFunction(V_d)

d_old = fem.Function(V_d)

# --- Elastic energy ---
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda * ufl.tr(epsilon(u)) * Identity(dim) + 2 * mu * epsilon(u)

def psi_plus(u):
    e = epsilon(u)
    e_plus = ufl.tr(e) / 3
    return 0.5 * lmbda * ufl.tr(e)**2 + mu * inner(e, e)

# Total energy functional
elastic_energy = (1 - d)**2 * psi_plus(u) * dx
fracture_energy = Gc / (2 * ell) * d**2 * dx + Gc * ell / 2 * dot(grad(d), grad(d)) * dx

E_u = elastic_energy
E_d = derivative(elastic_energy, d, w) + derivative(fracture_energy, d, w)

F_u = ufl.derivative(E_u, u, v)
J_u = ufl.derivative(F_u, u)

# --- Dirichlet BCs ---
def left(x): return np.isclose(x[0], 0.0)
def right(x): return np.isclose(x[0], length)
def bottom(x): return np.isclose(x[1], 0.0)

dofs_left = fem.locate_dofs_geometrical(V_u.sub(0), left)
dofs_right = fem.locate_dofs_geometrical(V_u.sub(0), right)
dofs_bottom_y = fem.locate_dofs_geometrical(V_u.sub(1), bottom)

u_L = fem.Function(fem.FunctionSpace(domain, ("Lagrange", 1)))
u_L.interpolate(lambda x: 0.1 * x[0])

zero = fem.Function(fem.FunctionSpace(domain, ("Lagrange", 1)))
zero.interpolate(lambda x: np.zeros_like(x[0]))

bcs_u = [
    fem.dirichletbc(zero, dofs_left, V_u.sub(0)),
    fem.dirichletbc(u_L, dofs_right, V_u.sub(0)),
    fem.dirichletbc(zero, dofs_bottom_y, V_u.sub(1))
]

# --- Solvers ---
problem_u = fem.petsc.NonlinearProblem(F_u, u, bcs_u, J_u)
solver_u = fem.petsc.NewtonSolver(domain.comm, problem_u)

# Damage solver (linear)
a_d = fem.form(Gc * ell * inner(grad(d), grad(w)) * dx + Gc / ell * d * w * dx + 2 * (1 - d_old) * psi_plus(u) * w * dx)
L_d = fem.form(Gc / ell * w * dx)
A_d = fem.petsc.assemble_matrix(a_d)
A_d.assemble()
b_d = fem.petsc.create_vector(L_d)

# Initial conditions
d.x.array[:] = 0.0
d_old.x.array[:] = 0.0

# --- Time loop ---
num_steps = 10
for step in range(num_steps):
    solver_u.solve(u)
    u.x.scatter_forward()

    # Solve for damage
    d_old.x.array[:] = d.x.array
    with b_d.localForm() as loc: loc.set(0)
    fem.petsc.assemble_vector(b_d, L_d)
    fem.petsc.apply_lifting(b_d, [a_d], [[fem.dirichletbc(zero, fem.locate_dofs_geometrical(V_d, bottom))]])
    b_d.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b_d, [fem.dirichletbc(zero, fem.locate_dofs_geometrical(V_d, bottom))])
    solver_d = PETSc.KSP().create(domain.comm)
    solver_d.setOperators(A_d)
    solver_d.setType("cg")
    solver_d.getPC().setType("jacobi")
    solver_d.solve(b_d, d.vector)
    d.x.scatter_forward()

    # --- Visualization ---
    print(f"Step {step + 1}: max damage = {d.x.array.max():.3f}")
    if step == num_steps - 1:
        topology, cell_types, geometry = vtk_mesh(domain, domain.topology.dim)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["damage"] = d.x.array
        grid.point_data["ux"] = u.x.array.reshape((-1, 2))[:, 0]
        grid.point_data["uy"] = u.x.array.reshape((-1, 2))[:, 1]

        p = pv.Plotter()
        p.add_mesh(grid, scalars="damage", cmap="plasma", show_edges=True)
        p.view_xy()
        p.show()
