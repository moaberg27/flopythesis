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

# Polar angle θ: upper hemisphere only (0° to 90°), step 15°
# Each direction n is paired with its antipodal -n (the negative direction),
# so the upper hemisphere + lower hemisphere together cover the full sphere.
rotations_polar   = np.arange(0, 91, 15)    # 7 values: 0, 15, ..., 90

# Azimuth φ: full 360° around Z — needed to cover all directions in the upper hemisphere
rotations_azimuth = np.arange(0, 360, 15)   # 24 values: 0, 15, ..., 345

def compute_conductivity_points_3d(k, rotations_azimuth, rotations_polar):
    """Sample conductivity over the full sphere using +/- direction pairs.

    For each (theta, phi) in the upper hemisphere the unit direction vector is:
        n = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    Both the positive direction (n) and the negative direction (-n) are sampled,
    covering the full sphere without redundancy (one hemisphere per sign).
    """
    points = []
    for theta_deg in rotations_polar:
        theta_rad = np.radians(theta_deg)
        # At the north pole (theta=0) all azimuths collapse to the same axis — sample once
        phi_list = [0] if theta_deg == 0 else rotations_azimuth
        for phi_deg in phi_list:
            phi_rad = np.radians(phi_deg)
            nx = np.sin(theta_rad) * np.cos(phi_rad)
            ny = np.sin(theta_rad) * np.sin(phi_rad)
            nz = np.cos(theta_rad)
            # Positive direction (upper hemisphere)
            k_pos = calc_conductivity(k, theta_deg, direction='n')
            points.append([ k_pos * nx,  k_pos * ny,  k_pos * nz])
            # Negative direction / antipodal (lower hemisphere)
            k_neg = calc_conductivity(k, theta_deg, direction='n')
            points.append([-k_neg * nx, -k_neg * ny, -k_neg * nz])
    return np.array(points)

all_points = compute_conductivity_points_3d(1, rotations_azimuth, rotations_polar)
print(f"Sampled {len(all_points)} conductivity points in 3D:")


# 3D scatter plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*all_points.T, c='steelblue', marker='o', s=10, label='Sampled directions')
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
# all_points already contains the full spherical sampling

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

# all_points already contains the full spherical sampling

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