import numpy as np
import pyvista as pv
 
# Example nodes (z=0 for 2D mesh)
points = np.array([
    [0, 0, 0],  # 0
    [1, 0, 0],  # 1
    [2, 0, 0],  # 2
    [0, 1, 0],  # 3
    [1, 1, 0],  # 4
    [2, 1, 0],  # 5
])
 
# Example quads (each row = one element's node indices)
quads = np.array([
    [0, 1, 4, 3],  # First quad
    [1, 2, 5, 4],  # Second quad
])
 
# VTK faces format: [npts, id1, id2, id3, id4] repeated
faces = np.hstack([[4, *quad] for quad in quads])
 
# Create PolyData
mesh = pv.PolyData(points, faces)
 
# Example scalar values
temperature = np.array([10, 12, 14, 16, 18, 20])
mesh.point_data["temperature"] = temperature

# Save to VTP
mesh.save("multi_quad_mesh.vtp")
print("Saved multi_quad_mesh.vtp")