import meshio

msh = meshio.read("notched_plate_2.msh")

# Ensure 2D points
points_2d = msh.points[:, :2]

# Triangles (domain)
tri_mesh = meshio.Mesh(
    points=points_2d,
    cells=[("triangle", msh.cells_dict["triangle"])]
)
meshio.write("notched_plate_mesh_2.xdmf", tri_mesh)

# Facets (boundaries)
if "line" in msh.cells_dict and "line" in msh.cell_data_dict["gmsh:physical"]:
    facet_mesh = meshio.Mesh(
        points=points_2d,
        cells=[("line", msh.cells_dict["line"])],
        cell_data={"name_to_read": [msh.cell_data_dict["gmsh:physical"]["line"]]}
    )
    meshio.write("notched_plate_facet_region_2.xdmf", facet_mesh)
else:
    print("Warning: no 'line' tags found in msh file.")
