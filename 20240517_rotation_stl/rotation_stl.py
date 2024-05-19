import numpy as np
from stl import mesh
from scipy.spatial.transform import Rotation as R

def rotate_stl(stl_mesh, angular_velocity, steps=360):
    """
    Rotate the STL mesh according to the given angular velocity.
    
    Parameters:
    stl_mesh (mesh.Mesh): The STL mesh object to be rotated.
    angular_velocity (np.array): The angular velocity vector (in degrees per step).
    steps (int): Number of steps to complete one full rotation (default is 360).
    
    Returns:
    mesh.Mesh: The rotated STL mesh.
    """
    # Calculate the rotation per step
    rotation_per_step = angular_velocity / steps

    # Create a copy of the mesh to rotate
    rotated_mesh = mesh.Mesh(np.copy(stl_mesh.data))

    for i in range(steps):
        # Calculate the rotation matrix for the current step
        r = R.from_euler('xyz', rotation_per_step * i, degrees=True)
        rotation_matrix = r.as_matrix()

        # Apply the rotation to each vertex in the mesh
        for j, v in enumerate(rotated_mesh.vectors):
            rotated_mesh.vectors[j] = np.dot(v, rotation_matrix)

    return rotated_mesh

# Load the STL file
your_mesh = mesh.Mesh.from_file('path/to/your/file.stl')

# Define angular velocity (e.g., [1, 1, 1] degrees per step for x, y, z axes)
angular_velocity = np.array([1, 1, 1])

# Rotate the mesh
rotated_mesh = rotate_stl(your_mesh, angular_velocity)

# Save the rotated mesh to a new STL file
rotated_mesh.save('path/to/your/rotated_file.stl')
