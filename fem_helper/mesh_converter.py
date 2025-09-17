import meshio

msh = meshio.read("notched_plate_mixed.msh")

points_2d = msh.points[:, :2]

# Write mesh (triangles only)
triangle_mesh = meshio.Mesh(
    points=points_2d,
    cells=[("triangle", msh.cells_dict["triangle"])],
    cell_data={
        "name_to_read": [msh.cell_data_dict["gmsh:physical"]["triangle"]]
    }
)
triangle_mesh.write("notched_plate_mixed.xdmf")

# Write facet tags (lines only)
if "line" in msh.cells_dict:
    facet_mesh = meshio.Mesh(
        points=points_2d,
        cells=[("line", msh.cells_dict["line"])],
        cell_data={
            "name_to_read": [msh.cell_data_dict["gmsh:physical"]["line"]]
        }
    )
    facet_mesh.write("notched_plate_mixed_facet_region.xdmf")
else:
    print("No line cells found for facets")
