import pyvista as pv

# Load the .vtp file
mesh = pv.read("/home/sces201/Afeera/MeshGraphNet/scripts/shell_mgn/size1_1000_epoch_results/shell_graph_13.vtp")

# Inspect contents
print(mesh)

# Plot
mesh.plot(show_edges=True)
