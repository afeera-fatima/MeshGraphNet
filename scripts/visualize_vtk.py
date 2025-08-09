import pyvista as pv

# Load the .vtp file
mesh = pv.read("/home/sces201/Afeera/ML_Task/scripts/shell_mgn/low_precision_results/shell_graph_93.vtp")

# Inspect contents
print(mesh)

# Plot
mesh.plot(show_edges=True)
