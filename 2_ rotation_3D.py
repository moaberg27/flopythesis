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
        angle_rad = np.radians(angle) # primary angle for rotation
        angle_180 = np.radians(angle + 180) # angle for opposite direction in x
        angle_90 = np.radians(angle + 90) # angle for y-direction
        angle_270 = np.radians(angle + 270) # angle for opposite direction in y 

        # Rotation in the XY-plane (z = 0)
        kx = calc_conductivity(k, angle, direction='x') # conductivity in the x-direction
        points_x_pos.append([kx * np.cos(angle_rad), kx * np.sin(angle_rad), 0.0]) # z = 0 for XY-plane

        kx_neg = calc_conductivity(k, angle + 180, direction='x') # conductivity in the opposite direction for x
        points_x_neg.append([kx_neg * np.cos(angle_180), kx_neg * np.sin(angle_180), 0.0]) # z = 0 for XY-plane

        ky = calc_conductivity(k, angle, direction='y') # conductivity in the y-direction
        points_y_pos.append([ky * np.cos(angle_90), ky * np.sin(angle_90), 0.0]) # z = 0 for XY-plane

        ky_neg = calc_conductivity(k, angle + 270, direction='y') # conductivity in the opposite direction for y
        points_y_neg.append([ky_neg * np.cos(angle_270), ky_neg * np.sin(angle_270), 0.0]) # z = 0 for XY-plane

        # Rotation in the XZ-plane (y = 0)
        # Skip +x/-x at angle=0 (already measured in XY-plane)
        if angle != 0: # Skip +x/-x at angle=0 (already measured in XY-plane)
            kxz = calc_conductivity(k, angle, direction='x') # conductivity in the x-direction 
            points_x_pos.append([kxz * np.cos(angle_rad), 0.0, kxz * np.sin(angle_rad)])

            kxz_neg = calc_conductivity(k, angle + 180, direction='x') # conductivity in the opposite direction for x
            points_x_neg.append([kxz_neg * np.cos(angle_180), 0.0, kxz_neg * np.sin(angle_180)])

        kz_xz = calc_conductivity(k, angle, direction='z') # conductivity in the z-direction
        points_z_pos.append([kz_xz * np.cos(angle_90), 0.0, kz_xz * np.sin(angle_90)])

        kz_xz_neg = calc_conductivity(k, angle + 270, direction='z') # conductivity in the opposite direction for z
        points_z_neg.append([kz_xz_neg * np.cos(angle_270), 0.0, kz_xz_neg * np.sin(angle_270)])

        # Rotation in the YZ-plane (x = 0)
        # Skip +y/-y at angle=0 (already measured in XY-plane)
        # Skip +z/-z at angle=0 (already measured in XZ-plane)
        if angle != 0:
            kyz = calc_conductivity(k, angle, direction='y') # conductivity in the y-direction
            points_y_pos.append([0.0, kyz * np.cos(angle_rad), kyz * np.sin(angle_rad)])

            kyz_neg = calc_conductivity(k, angle + 180, direction='y') # conductivity in the opposite direction for y
            points_y_neg.append([0.0, kyz_neg * np.cos(angle_180), kyz_neg * np.sin(angle_180)])

            kz_yz = calc_conductivity(k, angle, direction='z') # conductivity in the z-direction
            points_z_pos.append([0.0, kz_yz * np.cos(angle_90), kz_yz * np.sin(angle_90)])

            kz_yz_neg = calc_conductivity(k, angle + 270, direction='z') # conductivity in the opposite direction for z
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


# 3D surface from conductivity points using ConvexHull + PyVista
all_points = np.vstack([
    points_x_pos, points_x_neg,
    points_y_pos, points_y_neg,
    points_z_pos, points_z_neg,
])

hull = ConvexHull(all_points)
faces = hull.simplices  # (n_faces, 3) triangle indices

# PyVista requires faces as [3, v0, v1, v2, 3, v0, v1, v2, ...]
pv_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()
surface = pv.PolyData(all_points, pv_faces)

plotter_hull = pv.Plotter()
plotter_hull.add_mesh(surface, color='cyan', opacity=0.5, show_edges=True, label='Convex Hull Surface')
plotter_hull.add_points(all_points, color='red', point_size=10, label='Measurements')
plotter_hull.add_legend()
plotter_hull.add_axes()
plotter_hull.show()

# Fit 3D conductivity tensor from directional measurements
# Each measurement satisfies:  k = n^T K n
# k = Kxx*nx² + Kyy*ny² + Kzz*nz² + 2Kxy*nx*ny + 2Kxz*nx*nz + 2Kyz*ny*nz

# Collect all measurement points and their magnitudes
all_points = np.vstack([
    points_x_pos, points_x_neg,
    points_y_pos, points_y_neg,
    points_z_pos, points_z_neg,
])

k_meas = np.linalg.norm(all_points, axis=1)  # measured conductivity magnitudes
directions = all_points / k_meas[:, np.newaxis]  # unit direction vectors (nx, ny, nz)

nx, ny, nz = directions[:, 0], directions[:, 1], directions[:, 2]

# Build design matrix A so that A @ [Kxx, Kxy, Kxz, Kyy, Kyz, Kzz] = k_meas
A = np.column_stack([
    nx**2,
    2 * nx * ny,
    2 * nx * nz,
    ny**2,
    2 * ny * nz,
    nz**2,
])

result, _, _, _ = np.linalg.lstsq(A, k_meas, rcond=None)
Kxx, Kxy, Kxz, Kyy, Kyz, Kzz = result

K_tensor = np.array([
    [Kxx, Kxy, Kxz],
    [Kxy, Kyy, Kyz],
    [Kxz, Kyz, Kzz],
])

# Principal values and orientation via eigendecomposition
eigvals, eigvecs = np.linalg.eigh(K_tensor)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

# RMSE: how well the points fit an ellipsoid (0 = perfect continuum)
k_pred = A @ result
rmse = np.sqrt(np.mean((k_meas - k_pred)**2))

print("\nFitted 3D conductivity tensor")
print(f"  K = [[{Kxx:.4e}, {Kxy:.4e}, {Kxz:.4e}],")
print(f"       [{Kxy:.4e}, {Kyy:.4e}, {Kyz:.4e}],")
print(f"       [{Kxz:.4e}, {Kyz:.4e}, {Kzz:.4e}]]")
print(f"  Principal values: k1 = {eigvals[0]:.4e}, k2 = {eigvals[1]:.4e}, k3 = {eigvals[2]:.4e}")
print(f"  Principal axes:")

# Show the tensor in the principal axes (diagonalized)
print("K tensor in principal axes (diagonalized):")
print(np.diag(eigvals))
for i in range(3):
    print(f"    v{i+1} = [{eigvecs[0,i]:.4f}, {eigvecs[1,i]:.4f}, {eigvecs[2,i]:.4f}]")
print(f"  RMSE residual:    {rmse:.4e}  (0 = perfect ellipsoid)")


# Plot the fitted ellipsoid with PyVista
# Create unit sphere and transform to ellipsoid aligned with principal axes
sphere = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
sphere_pts = np.array(sphere.points)

# Transform:  ellipsoid_pt = eigvecs @ diag(eigvals) @ sphere_pt
ellipsoid_pts = (eigvecs @ np.diag(eigvals) @ sphere_pts.T).T
ellipsoid = sphere.copy()
ellipsoid.points = ellipsoid_pts

# Plot
plotter = pv.Plotter()
plotter.add_mesh(ellipsoid, color='lightblue', opacity=0.4, label='Fitted Ellipsoid')
plotter.add_points(all_points, color='red', point_size=8, label='Measurements')

# Draw principal axes with labels
labels = ['k1 (1st)', 'k2 (2nd)', 'k3 (3rd)']
for i, color in enumerate(['red', 'green', 'blue']):
    axis = eigvecs[:, i] * eigvals[i]
    plotter.add_arrows(np.zeros((1, 3)), axis.reshape(1, 3), color=color, mag=1)
    plotter.add_point_labels(
        axis.reshape(1, 3),
        [f"{labels[i]} = {eigvals[i]:.4f}"],
        font_size=12,
        text_color=color,
        bold=True,
    )

plotter.add_legend()
plotter.add_axes()
plotter.show()


