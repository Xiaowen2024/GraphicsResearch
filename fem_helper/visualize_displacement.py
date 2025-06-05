import numpy as np
import pyvista as pv

from dolfinx import fem, io, plot
from mpi4py import MPI
from basix.ufl import element

# === Read mesh from XDMF ===
with io.XDMFFile(MPI.COMM_WORLD, "displacement.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, tdim)

# === Recreate function space and load displacement ===
el = element("Lagrange", domain.ufl_cell().cellname(), 1, shape=(2,))
V = fem.functionspace(domain, el)
uh = fem.Function(V)
uh.name = "Displacement"
uh.read_information("displacement.h5", "/Displacement")

# === Extract VTK data using plot.vtk_mesh ===
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# === Add displacement vectors to grid ===
u_array = uh.x.array.reshape((-1, 2))
u_3d = np.hstack([u_array, np.zeros((u_array.shape[0], 1))])  # pad (N,2) -> (N,3)
grid.point_data["Displacement"] = u_3d

# === Visualize using PyVista ===
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.add_arrows(grid.points, u_3d, mag=5.0, label="Displacement")
plotter.add_axes()
plotter.view_xy()
plotter.show()
