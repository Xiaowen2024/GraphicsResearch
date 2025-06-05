# import meshio

# msh = meshio.read("notched_plate.msh")

# # --- Triangles (domain) ---
# triangle_cells = [c for c in msh.cells if c.type == "triangle"]
# triangle_data = [msh.cell_data_dict["gmsh:physical"][c.type] for c in msh.cells if c.type == "triangle"]
# meshio.write("notched_plate_mesh.xdmf", meshio.Mesh(
#     points=msh.points,
#     cells=triangle_cells,
#     cell_data={"name_to_read": triangle_data}
# ))

# # --- Lines (facet tags) ---
# line_cells = [c for c in msh.cells if c.type == "line"]
# line_data = [msh.cell_data_dict["gmsh:physical"][c.type] for c in msh.cells if c.type == "line"]

# meshio.write("notched_plate_facet_region.xdmf", meshio.Mesh(
#     points=msh.points,
#     cells=[("line", line_cells[0].data)],  # Make sure data is structured correctly
#     cell_data={"name_to_read": [line_data[0]]}
# ))

import meshio

msh = meshio.read("notched_plate.msh")

# Ensure 2D points
points_2d = msh.points[:, :2]

# Triangles (domain)
tri_mesh = meshio.Mesh(
    points=points_2d,
    cells=[("triangle", msh.cells_dict["triangle"])]
)
meshio.write("notched_plate_mesh.xdmf", tri_mesh)

# Facets (boundaries)
if "line" in msh.cells_dict and "line" in msh.cell_data_dict["gmsh:physical"]:
    facet_mesh = meshio.Mesh(
        points=points_2d,
        cells=[("line", msh.cells_dict["line"])],
        cell_data={"name_to_read": [msh.cell_data_dict["gmsh:physical"]["line"]]}
    )
    meshio.write("notched_plate_facet_region.xdmf", facet_mesh)
else:
    print("Warning: no 'line' tags found in msh file.")
