import numpy as np
from stl import mesh

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def calculate_incidence_angle(normal, light_direction):
    # Normalize the vectors
    normal = normalize(normal)
    light_direction = normalize(light_direction)
    
    # Calculate the dot product
    dot_product = np.dot(normal, light_direction)
    
    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# Load the STL file
your_mesh = mesh.Mesh.from_file('./egg.stl')

# Define light direction (e.g., from above)
light_direction = np.array([0, 0, -1])

# Iterate through each face of the mesh
for i, facet in enumerate(your_mesh.normals):
    incidence_angle = calculate_incidence_angle(facet, light_direction)
    print(f'Facet {i}: Incidence Angle = {incidence_angle:.2f} degrees')
