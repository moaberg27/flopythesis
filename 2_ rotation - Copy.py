# Here we build the rotation of the box in 3D
import numpy as np
from scipy.spatial import ConvexHull
import pyvista as pv

def calc_conductivity(k, theta, direction):
    
    # Convert theta to radians
    theta_radians = np.radians(theta)
    
    # Calculate the effective conductivity 
    k_eff = k * (np.cos(theta_radians)**2 + np.sin(theta_radians)**2) * (1-np.random.rand()/5)  # Adding some randomness to simulate variability
    
    return k_eff

rotations = [0, 15, 30, 45, 60, 75]

def compute_conductivity_points_3d(k, rotations):
    points_x_pos = []
    points_y_pos = []
    points_x_neg = []
    points_y_neg = []
    points_z_pos = []
    points_z_neg = []

    for angle in rotations:
        angle_rad = np.radians(angle)
        angle_180 = np.radians(angle + 180)
        angle_90 = np.radians(angle + 90)
        angle_270 = np.radians(angle + 270)

        # Rotation in the xy-plane (z = 0) ---
        # 2D ellipse
        kx = calc_conductivity(k, angle, direction='x') # conductivity in the x-direction
        points_x_pos.append([kx * np.cos(angle_rad), kx * np.sin(angle_rad), 0.0]) # 

        kx_neg = calc_conductivity(k, angle + 180, direction='x') # conductivity in the opposite direction for x
        points_x_neg.append([kx_neg * np.cos(angle_180), kx_neg * np.sin(angle_180), 0.0]) 

        ky = calc_conductivity(k, angle, direction='y') # conductivity in the y-direction
        points_y_pos.append([ky * np.cos(angle_90), ky * np.sin(angle_90), 0.0])

        ky_neg = calc_conductivity(k, angle + 270, direction='y') # conductivity in the opposite direction for y
        points_y_neg.append([ky_neg * np.cos(angle_270), ky_neg * np.sin(angle_270), 0.0])

        # Rotation in the xz-plane (y = 0)
        # x-direction rotated toward z
        kxz = calc_conductivity(k, angle, direction='x') 
        points_x_pos.append([kxz * np.cos(angle_rad), 0.0, kxz * np.sin(angle_rad)])

        kxz_neg = calc_conductivity(k, angle + 180, direction='x') 
        points_x_neg.append([kxz_neg * np.cos(angle_180), 0.0, kxz_neg * np.sin(angle_180)])

        # z-direction in xz-plane (perpendicular to x)
        kz_xz = calc_conductivity(k, angle, direction='z')
        points_z_pos.append([kz_xz * np.cos(angle_90), 0.0, kz_xz * np.sin(angle_90)])

        kz_xz_neg = calc_conductivity(k, angle + 270, direction='z')
        points_z_neg.append([kz_xz_neg * np.cos(angle_270), 0.0, kz_xz_neg * np.sin(angle_270)])

        # Rotation in the yz-plane (x = 0)
        # y-direction rotated toward z
        kyz = calc_conductivity(k, angle, direction='y')
        points_y_pos.append([0.0, kyz * np.cos(angle_rad), kyz * np.sin(angle_rad)])

        kyz_neg = calc_conductivity(k, angle + 180, direction='y')
        points_y_neg.append([0.0, kyz_neg * np.cos(angle_180), kyz_neg * np.sin(angle_180)])

        # z-direction in yz-plane (perpendicular to y)
        kz_yz = calc_conductivity(k, angle, direction='z')
        points_z_pos.append([0.0, kz_yz * np.cos(angle_90), kz_yz * np.sin(angle_90)])

        kz_yz_neg = calc_conductivity(k, angle + 270, direction='z')
        points_z_neg.append([0.0, kz_yz_neg * np.cos(angle_270), kz_yz_neg * np.sin(angle_270)])

    return points_x_pos, points_y_pos, points_x_neg, points_y_neg, points_z_pos, points_z_neg

points_x_pos, points_y_pos, points_x_neg, points_y_neg, points_z_pos, points_z_neg = compute_conductivity_points_3d(1, rotations)

# Convert to arrays for plotting
points_x_pos = np.array(points_x_pos)
points_y_pos = np.array(points_y_pos)
points_x_neg = np.array(points_x_neg)
points_y_neg = np.array(points_y_neg)
points_z_pos = np.array(points_z_pos)
points_z_neg = np.array(points_z_neg)

# 3D scatter plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*points_x_pos.T, c='red', marker='o', label='+x')
ax.scatter(*points_x_neg.T, c='darkred', marker='o', label='-x')
ax.scatter(*points_y_pos.T, c='blue', marker='s', label='+y')
ax.scatter(*points_y_neg.T, c='darkblue', marker='s', label='-y')
ax.scatter(*points_z_pos.T, c='green', marker='^', label='+z')
ax.scatter(*points_z_neg.T, c='darkgreen', marker='^', label='-z')
ax.set_xlabel('Conductivity x')
ax.set_ylabel('Conductivity y')
ax.set_zlabel('Conductivity z')
ax.set_title('3D Effective Conductivity Ellipsoid')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.legend()
plt.show()

# PyVista surface mesh 
all_points = np.vstack([points_x_pos, points_x_neg, points_y_pos, points_y_neg, points_z_pos, points_z_neg])

# Remove duplicate points (z points overlap since x,y = 0 for all)
all_points = np.unique(all_points, axis=0)

def build_nodes_and_faces(nodes):
    """Triangulate the surface using a convex hull and return PyVista-compatible faces."""
    hull = ConvexHull(nodes)
    faces = hull.simplices
    faces_flat = np.hstack([[3, *f] for f in faces]).astype(np.int64)
    return faces, faces_flat

faces, faces_flat = build_nodes_and_faces(all_points)
print(f"Number of nodes: {len(all_points)}")
print(f"Number of triangular faces: {len(faces)}")

mesh = pv.PolyData(all_points, faces_flat)
plotter = pv.Plotter(window_size=(1200, 800))
plotter.add_mesh(mesh, color="lightsteelblue", opacity=0.45, show_edges=True)
plotter.add_points(all_points, color="crimson", point_size=12, render_points_as_spheres=True)
plotter.add_axes()
plotter.add_title("Conductivity Ellipsoid (Nodes + Faces)")
plotter.show()
