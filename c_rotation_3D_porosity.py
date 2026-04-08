# Here we build the rotation of the box in 3D
import numpy as np
from scipy.spatial import ConvexHull
import pyvista as pv

rotations = [0, 15, 30, 45, 60, 75]


def rotation_matrix_zxy(z_deg, x_deg, y_deg):
    """Local-to-global rotation matrix for sequence z -> x -> y."""
    z_r, x_r, y_r = np.radians(z_deg), np.radians(x_deg), np.radians(y_deg)
    cz, sz = np.cos(z_r), np.sin(z_r)
    cx, sx = np.cos(x_r), np.sin(x_r)
    cy, sy = np.cos(y_r), np.sin(y_r)
    rz = np.array([[cz, -sz, 0.], [sz,  cz, 0.], [0., 0., 1.]])
    rx = np.array([[1., 0.,  0.], [0.,  cx, -sx], [0., sx, cx]])
    ry = np.array([[cy, 0., sy],  [0.,  1., 0.],  [-sy, 0., cy]])
    return ry @ rx @ rz


# Underlying "true" anisotropic tensor
np.random.seed(42)
true_principal = np.array([6.0, 3.0, 1.0])
r_true = rotation_matrix_zxy(z_deg=25.0, x_deg=35.0, y_deg=10.0)
k_true = r_true @ np.diag(true_principal) @ r_true.T

# True porosity tensor (principal porosities and orientation — independent of K)
# Represents directional fracture storage/connectivity
n_principal = np.array([0.05, 0.02, 0.005])   # n1 > n2 > n3  [-]
r_n_true = rotation_matrix_zxy(z_deg=15.0, x_deg=20.0, y_deg=5.0)
n_true = r_n_true @ np.diag(n_principal) @ r_n_true.T

DOMAIN_LENGTH = 1.0  # representative domain length [m] for particle tracking


def directional_k(k_tensor, direction_vec):
    """Directional conductivity k(n) = n^T K n for unit direction n."""
    n = np.asarray(direction_vec, dtype=float)
    n = n / np.linalg.norm(n)
    return float(n @ k_tensor @ n)


def directional_n(n_tensor, direction_vec):
    """Directional porosity η(d) = d^T N d for unit direction d."""
    d = np.asarray(direction_vec, dtype=float)
    d = d / np.linalg.norm(d)
    return float(d @ n_tensor @ d)


def mock_dfn(rotation_deg, noise=0.10):
    """Compute directional K for local +/-x, +/-y, +/-z at one 3D rotation."""
    r = rotation_matrix_zxy(*rotation_deg)
    ex, ey, ez = r[:, 0], r[:, 1], r[:, 2]

    def noisy_k(vec):
        k = directional_k(k_true, vec)
        k = k * (1.0 + noise * np.random.randn())
        return max(k, 1e-8)

    return (
        noisy_k( ex), noisy_k(-ex),
        noisy_k( ey), noisy_k(-ey),
        noisy_k( ez), noisy_k(-ez),
    )


def mock_particle_tracking(direction_vec, k_dir, noise=0.10):
    """Estimate directional porosity from simulated particle tracking.

    Formula:  n = Q / V  =  q / (L/T)  =  q * T / L
      q = K_dir * i   Darcy flux (unit hydraulic gradient i = 1)
      L               pathline length measured along the particle track
      T               travel time (mean/median over the particle ensemble)

    The true travel time follows from the advection equation:
        T_true = L * n_dir / q
    Noise is added to both L (tortuosity) and T (measurement scatter).
    """
    q = k_dir   # Darcy flux with unit hydraulic gradient
    n_dir = max(directional_n(n_true, direction_vec), 1e-8)

    # Pathline length: domain length + small tortuosity variability
    L_meas = DOMAIN_LENGTH * (1.0 + 0.05 * abs(np.random.randn()))

    # True travel time from advection equation: T = L * n / q
    T_true = L_meas * n_dir / q

    # Noisy travel time (particle tracking measurement uncertainty)
    T_meas = T_true * (1.0 + noise * np.random.randn())
    T_meas = max(T_meas, 1e-12)

    # Recover porosity: n = q * T / L
    return max(q * T_meas / L_meas, 1e-8)


def compute_conductivity_points_3d(rotations):
    """Sample all 6^3 = 216 Euler-angle orientations (z, x, y) x 6 directions = 1296 points.

    For each orientation the rotation matrix maps the three local axes into
    global coordinates. Covers the full 3D directional space.
    Returns both K (conductivity) and n (porosity from particle tracking) vectors.
    """
    points_x_pos, points_x_neg = [], []
    points_y_pos, points_y_neg = [], []
    points_z_pos, points_z_neg = [], []
    n_x_pos, n_x_neg = [], []
    n_y_pos, n_y_neg = [], []
    n_z_pos, n_z_neg = [], []

    for z_deg in rotations:
        for x_deg in rotations:
            for y_deg in rotations:
                R = rotation_matrix_zxy(z_deg, x_deg, y_deg)

                # Local unit vectors expressed in the global frame
                ex =  R[:, 0]   # local +x direction
                ey =  R[:, 1]   # local +y direction
                ez =  R[:, 2]   # local +z direction

                kxp, kxn, kyp, kyn, kzp, kzn = mock_dfn((z_deg, x_deg, y_deg))

                # Conductivity vectors: k * direction
                points_x_pos.append(kxp *  ex)
                points_x_neg.append(kxn * -ex)
                points_y_pos.append(kyp *  ey)
                points_y_neg.append(kyn * -ey)
                points_z_pos.append(kzp *  ez)
                points_z_neg.append(kzn * -ez)

                # Porosity vectors: n = q*T/L, from particle tracking in each direction
                n_x_pos.append(mock_particle_tracking( ex, kxp) *  ex)
                n_x_neg.append(mock_particle_tracking(-ex, kxn) * -ex)
                n_y_pos.append(mock_particle_tracking( ey, kyp) *  ey)
                n_y_neg.append(mock_particle_tracking(-ey, kyn) * -ey)
                n_z_pos.append(mock_particle_tracking( ez, kzp) *  ez)
                n_z_neg.append(mock_particle_tracking(-ez, kzn) * -ez)

    return (points_x_pos, points_y_pos, points_x_neg, points_y_neg, points_z_pos, points_z_neg,
            n_x_pos, n_y_pos, n_x_neg, n_y_neg, n_z_pos, n_z_neg)


(points_x_pos, points_y_pos, points_x_neg, points_y_neg, points_z_pos, points_z_neg,
 n_x_pos, n_y_pos, n_x_neg, n_y_neg, n_z_pos, n_z_neg) = compute_conductivity_points_3d(rotations)
print(f"Total measurement points: {6 * len(rotations)**3}  (6 directions x {len(rotations)}^3 orientations)")

# Convert to arrays for plotting
points_x_pos = np.array(points_x_pos); points_x_neg = np.array(points_x_neg)
points_y_pos = np.array(points_y_pos); points_y_neg = np.array(points_y_neg)
points_z_pos = np.array(points_z_pos); points_z_neg = np.array(points_z_neg)
n_x_pos = np.array(n_x_pos); n_x_neg = np.array(n_x_neg)
n_y_pos = np.array(n_y_pos); n_y_neg = np.array(n_y_neg)
n_z_pos = np.array(n_z_pos); n_z_neg = np.array(n_z_neg)

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
ax.set_title('3D Effective Conductivity Ellipsoid (6³ orientations × 6 directions = 1296 pts)')
all_pts_tmp = np.vstack([points_x_pos, points_x_neg, points_y_pos, points_y_neg, points_z_pos, points_z_neg])
lim = np.max(np.abs(all_pts_tmp)) * 1.15
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_box_aspect([1, 1, 1])
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
rel_rmse = rmse / np.mean(k_meas) * 100.0
print(f"  RMSE residual:    {rmse:.4e}  (relative: {rel_rmse:.2f}%)")
if rel_rmse < 5:
    print("  --> Good ellipsoid fit: continuum assumption likely valid")
elif rel_rmse < 15:
    print("  --> Moderate fit: continuum assumption approximate")
else:
    print("  --> Poor fit: continuum assumption may NOT hold")


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


# ===========================================================================
# Porosity analysis
# n = Q/V = q / (L/T) = q * T / L
# Sampled the same way as K: 6^3 orientations x 6 directions = 1296 points
# ===========================================================================

all_n_points = np.vstack([n_x_pos, n_x_neg, n_y_pos, n_y_neg, n_z_pos, n_z_neg])

# --- Porosity scatter plot ---
fig_n = plt.figure(figsize=(8, 8))
ax_n = fig_n.add_subplot(111, projection='3d')
ax_n.scatter(*n_x_pos.T, c='red',       marker='o', label='+x')
ax_n.scatter(*n_x_neg.T, c='darkred',   marker='o', label='-x')
ax_n.scatter(*n_y_pos.T, c='blue',      marker='s', label='+y')
ax_n.scatter(*n_y_neg.T, c='darkblue',  marker='s', label='-y')
ax_n.scatter(*n_z_pos.T, c='green',     marker='^', label='+z')
ax_n.scatter(*n_z_neg.T, c='darkgreen', marker='^', label='-z')
ax_n.set_xlabel('Porosity x')
ax_n.set_ylabel('Porosity y')
ax_n.set_zlabel('Porosity z')
ax_n.set_title('3D Effective Porosity Ellipsoid (n = q·T/L, 1296 pts)')
lim_n = np.max(np.abs(all_n_points)) * 1.15
ax_n.set_xlim(-lim_n, lim_n)
ax_n.set_ylim(-lim_n, lim_n)
ax_n.set_zlim(-lim_n, lim_n)
ax_n.set_box_aspect([1, 1, 1])
ax_n.legend()
plt.show()

# --- Fit symmetric 3D porosity tensor: η(d) = d^T N d ---
n_meas_vals = np.linalg.norm(all_n_points, axis=1)
n_dirs      = all_n_points / n_meas_vals[:, np.newaxis]
nxd, nyd, nzd = n_dirs[:, 0], n_dirs[:, 1], n_dirs[:, 2]

A_n = np.column_stack([
    nxd**2,
    2 * nxd * nyd,
    2 * nxd * nzd,
    nyd**2,
    2 * nyd * nzd,
    nzd**2,
])
res_n, _, _, _ = np.linalg.lstsq(A_n, n_meas_vals, rcond=None)
Nxx, Nxy, Nxz, Nyy, Nyz, Nzz = res_n

N_tensor = np.array([
    [Nxx, Nxy, Nxz],
    [Nxy, Nyy, Nyz],
    [Nxz, Nyz, Nzz],
])

n_eigvals, n_eigvecs = np.linalg.eigh(N_tensor)
n_order   = np.argsort(n_eigvals)[::-1]
n_eigvals = n_eigvals[n_order]
n_eigvecs = n_eigvecs[:, n_order]

n_pred = A_n @ res_n
n_rmse = np.sqrt(np.mean((n_meas_vals - n_pred) ** 2))

print("\nFitted 3D porosity tensor")
print(f"  N = [[{Nxx:.4e}, {Nxy:.4e}, {Nxz:.4e}],")
print(f"       [{Nxy:.4e}, {Nyy:.4e}, {Nyz:.4e}],")
print(f"       [{Nxz:.4e}, {Nyz:.4e}, {Nzz:.4e}]]")
print(f"  True  principal porosities: n1={n_principal[0]:.4f}, n2={n_principal[1]:.4f}, n3={n_principal[2]:.4f}")
print(f"  Fitted principal porosities: n1={n_eigvals[0]:.4e}, n2={n_eigvals[1]:.4e}, n3={n_eigvals[2]:.4e}")
n_rel_rmse = n_rmse / np.mean(n_meas_vals) * 100.0
print(f"  RMSE residual: {n_rmse:.4e}  (relative: {n_rel_rmse:.2f}%)")
if n_rel_rmse < 5:
    print("  --> Good ellipsoid fit: continuum assumption likely valid")
elif n_rel_rmse < 15:
    print("  --> Moderate fit: continuum assumption approximate")
else:
    print("  --> Poor fit: continuum assumption may NOT hold")

# --- PyVista: fitted porosity ellipsoid ---
n_eigvals_plot = np.clip(n_eigvals, 1e-8, None)
sphere_n = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
n_ellipsoid_pts = (n_eigvecs @ np.diag(n_eigvals_plot) @ np.array(sphere_n.points).T).T
n_ellipsoid = sphere_n.copy()
n_ellipsoid.points = n_ellipsoid_pts

plotter_n = pv.Plotter()
plotter_n.add_mesh(n_ellipsoid, color='lightgreen', opacity=0.4, label='Fitted Porosity Ellipsoid')
plotter_n.add_points(all_n_points, color='blue', point_size=8, label='Porosity measurements')
labels_n = ['n1 (1st)', 'n2 (2nd)', 'n3 (3rd)']
for i, color in enumerate(['red', 'green', 'blue']):
    axis_n = n_eigvecs[:, i] * n_eigvals_plot[i]
    plotter_n.add_arrows(np.zeros((1, 3)), axis_n.reshape(1, 3), color=color, mag=1)
    plotter_n.add_point_labels(
        axis_n.reshape(1, 3),
        [f"{labels_n[i]} = {n_eigvals[i]:.4e}"],
        font_size=12, text_color=color, bold=True,
    )
plotter_n.add_legend()
plotter_n.add_axes()
plotter_n.show()


