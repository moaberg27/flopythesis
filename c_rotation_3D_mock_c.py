# WORKFLOW:
#   1. precompute_directions()  — tells you how many DFN runs needed
#   2. split_directions()       — split into fit / validation sets
#   3. run_dfn_simulation()     — called once per unique direction
#   4. fit_conductivity_tensor()— least-squares fit from the k values
#   5. plotting                 — visualise the fitted ellipsoid

import numpy as np
from scipy.spatial import ConvexHull
import pyvista as pv
import matplotlib.pyplot as plt

# =====================================================================
# DFN SOLVER INTERFACE — replace this with your actual flow solver
# =====================================================================
# True principal values — ONLY used by the synthetic placeholder below.
# In real use, you don't know these (that's what you're solving for).
K_PRINCIPAL = np.array([3.0, 3.0, 3.0])   # kx, ky, kz

# This function simulates running a DFN flow simulation for a given rotation and face index, returning an effective conductivity.
#  
# In practice, replace the body of this function with a call to your actual DFN solver, passing the appropriate rotation and gradient axis parameters.

# Phi is the azimuthal rotation around Z, and theta is the elevation rotation around X. 

# The face_index determines which pair of faces (±X, ±Y, or ±Z) the pressure gradient is applied to in the local frame of the domain.
# If 0, the gradient is along local X (faces ±X); if 1, along local Y (faces ±Y); if 2, along local Z (faces ±Z).
def run_dfn_simulation(phi_deg, theta_deg, face_index):
    """
    Run a single DFN flow simulation and return the effective conductivity.

    Parameters
    ----------
    phi_deg : float
        Azimuth rotation of the domain around Z (degrees).
    theta_deg : float
        Elevation rotation of the domain around X (degrees).
    face_index : int
        Which face pair to impose the pressure gradient on:
          0 → gradient along local X (faces ±X)
          1 → gradient along local Y (faces ±Y)
          2 → gradient along local Z (faces ±Z)

    Returns
    -------
    k_eff : float
        Effective (equivalent) conductivity measured for this configuration.

    NOTE: This is a synthetic placeholder using k_eff = n^T K n + noise.
    Replace this entire function body with your real DFN solver call, e.g.:
        result = your_dfn_solver.run(domain, rotation=(phi_deg, theta_deg),
                                     gradient_axis=face_index)
        return result.equivalent_conductivity
    """
    # R is the rotation matrix that transforms local face normals to lab frame directions based on the specified phi and theta rotations.
    R = rotation_matrix('x', theta_deg) @ rotation_matrix('z', phi_deg)
    n_lab = R @ FACE_NORMALS_LOCAL[face_index] # takes a local axis and rotates it to the lab frame, giving the measurement direction in the lab frame
    k_eff = (K_PRINCIPAL[0] * n_lab[0]**2 +
             K_PRINCIPAL[1] * n_lab[1]**2 +
             K_PRINCIPAL[2] * n_lab[2]**2)
    noise = 1.0 - np.random.rand() / 5   # +-20 % variability
    return k_eff * noise


def rotation_matrix(axis, angle_deg):
    """Return the 3x3 rotation matrix for rotating `angle_deg` around `axis`."""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x': # rotation around X-axis
        return np.array([[1, 0,  0], 
                         [0, c, -s],
                         [0, s,  c]])
    elif axis == 'z': # rotation around Z-axis
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]])
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")


# 3 face normals only — opposite faces give redundant info since k(n) = k(-n)
FACE_NORMALS_LOCAL = np.array([
    [1, 0, 0], # local X face normal
    [0, 1, 0], # local Y face normal
    [0, 0, 1], # local Z face normal
], dtype=float)


def build_design_matrix(directions):
    """
    Build the least-squares design matrix for the symmetric tensor fit.

    Each row encodes:  k = Kxx*nx^2 + 2Kxy*nx*ny + 2Kxz*nx*nz + Kyy*ny^2 + 2Kyz*ny*nz + Kzz*nz^2
    so A @ [Kxx, Kxy, Kxz, Kyy, Kyz, Kzz] = k_meas
    """
    nx, ny, nz = directions[:, 0], directions[:, 1], directions[:, 2]
    return np.column_stack([
        nx**2,
        2 * nx * ny,
        2 * nx * nz,
        ny**2,
        2 * ny * nz,
        nz**2,
    ])


def precompute_directions(rotations):
    """
    PHASE 1 — Precompute unique measurement directions (no DFN cost).

    Rotates the box face normals through the phi/theta grid and deduplicates.
    Each unique direction is paired with the rotation state (phi, theta, face_index)
    needed to physically realise that measurement in the DFN.

    Returns
    -------
    directions : (N, 3) ndarray
        Unique lab-frame unit vectors.
    rotation_states : list of (phi_deg, theta_deg, face_index) tuples
        Which rotation + face produces each direction — pass this to your
        DFN solver so it knows how to orient the domain.
    """
    step = rotations[1] - rotations[0] # assumes uniform step size in the rotation grid
    phi_angles   = list(range(0, 360, step))    # full azimuth rotation around Z
    theta_angles = list(range(0, 180, step))    # full polar arc rotation around X

    seen = set() # to track unique directions (rounded to avoid floating-point issues)
    unique_directions = []        # list of unique lab-frame unit vectors
    rotation_states   = []        # maps each unique direction to its rotation

    for phi_deg in phi_angles:
        for theta_deg in theta_angles:
            R = rotation_matrix('x', theta_deg) @ rotation_matrix('z', phi_deg)
            for face_idx, n_local in enumerate(FACE_NORMALS_LOCAL):
                n_lab = R @ n_local # rotate the local face normal to get the measurement direction in the lab frame
                key = tuple(np.round(n_lab, 8)) # round to avoid floating-point issues when checking uniqueness
                if key not in seen:
                    seen.add(key)
                    unique_directions.append(n_lab)
                    rotation_states.append((phi_deg, theta_deg, face_idx))

    return np.array(unique_directions), rotation_states

# This split directions function is used to create a training set (fit_directions) 
# and a validation set (val_directions) from the precomputed unique directions. 
def split_directions(directions, rotation_states, val_fraction=0.2, seed=42):
    """
    Split precomputed directions into fit / validation sets.

    Parameters
    ----------
    directions : (N, 3) ndarray
    rotation_states : list of tuples
    val_fraction : float
    seed : int

    Returns
    -------
    fit_dirs, val_dirs, fit_states, val_states
    """
    n_total = len(directions)
    n_val   = max(1, int(n_total * val_fraction))

    rng     = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    fit_idx = indices[:-n_val]
    val_idx = indices[-n_val:]

    fit_states = [rotation_states[i] for i in fit_idx]
    val_states = [rotation_states[i] for i in val_idx]

    return directions[fit_idx], directions[val_idx], fit_states, val_states


# ===========================================================================
# PHASE 1 — precompute directions ( no DFN cost)
# ===========================================================================
rotations = [0, 15, 30, 45, 60, 75]
all_directions, all_rotation_states = precompute_directions(rotations)

print(f"Rotation grid step: {rotations[1]-rotations[0]}°")
print(f"Unique measurement directions: {len(all_directions)}")
print(f"  → this is the number of DFN simulations you will need\n")

fit_directions, val_directions, fit_states, val_states = split_directions(
    all_directions, all_rotation_states, val_fraction=0.2,
)
print(f"Fit directions:        {len(fit_directions)}")
print(f"Validation directions: {len(val_directions)}")

# --- Export to CSV ---
import csv

with open('unique_directions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['nx', 'ny', 'nz'])
    writer.writerows(all_directions)

with open('rotation_states.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['phi_deg', 'theta_deg', 'face_index'])
    writer.writerows(all_rotation_states)

print(f"\nExported 'unique_directions.csv' and 'rotation_states.csv'")

# ===========================================================================
# PHASE 2 — run DFN simulations (expensive — only for unique directions)
# ===========================================================================
fit_k = np.array([run_dfn_simulation(*state) for state in fit_states])
val_k = np.array([run_dfn_simulation(*state) for state in val_states])

# Build measurement points (for plotting)
fit_points = fit_directions * fit_k[:, np.newaxis]
val_points = val_directions * val_k[:, np.newaxis]
all_points = np.vstack([fit_points, val_points])


# --- Fit the 3D conductivity tensor ---
A_fit = build_design_matrix(fit_directions)
result, _, _, _ = np.linalg.lstsq(A_fit, fit_k, rcond=None)
Kxx, Kxy, Kxz, Kyy, Kyz, Kzz = result

K_tensor = np.array([
    [Kxx, Kxy, Kxz],
    [Kxy, Kyy, Kyz],
    [Kxz, Kyz, Kzz],
])


# --- Fit quality ---
fit_k_pred = A_fit @ result
rmse_fit = np.sqrt(np.mean((fit_k - fit_k_pred) ** 2))

A_val = build_design_matrix(val_directions)
val_k_pred = A_val @ result
rmse_val = np.sqrt(np.mean((val_k - val_k_pred) ** 2))

cond = np.linalg.cond(A_fit)


# --- Principal values and orientation ---
eigvals, eigvecs = np.linalg.eigh(K_tensor)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]


# --- Console report ---
print(f"\nSampling: {len(fit_directions)} fit + {len(val_directions)} validation directions")
print(f"Design matrix condition number: {cond:.2f}  (< 100 is good)")
print(f"\nFitted 3D conductivity tensor:")
print(f"  K = [[{Kxx:.4e}, {Kxy:.4e}, {Kxz:.4e}],")
print(f"       [{Kxy:.4e}, {Kyy:.4e}, {Kyz:.4e}],")
print(f"       [{Kxz:.4e}, {Kyz:.4e}, {Kzz:.4e}]]")
print(f"\n  Principal values: k1={eigvals[0]:.4f}, k2={eigvals[1]:.4f}, k3={eigvals[2]:.4f}")
# Remove the line below when using real DFN data (true values won't be known)
print(f"  (True values:     k1={K_PRINCIPAL[0]:.4f}, k2={K_PRINCIPAL[1]:.4f}, k3={K_PRINCIPAL[2]:.4f})")
print(f"\n  Principal axes:")
print("  K tensor in principal axes (diagonalized):")
print(f"  {np.diag(eigvals)}")
for i in range(3):
    print(f"    v{i+1} = [{eigvecs[0,i]:.4f}, {eigvecs[1,i]:.4f}, {eigvecs[2,i]:.4f}]")
print(f"\n  RMSE on fit directions:        {rmse_fit:.4e}")
print(f"  RMSE on held-out (validation): {rmse_val:.4e}  <- key quality indicator")


# --- 3D scatter plot ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(fit_points[:, 0], fit_points[:, 1], fit_points[:, 2],
           c='red', marker='o', s=10, label='Fit measurements')
ax.scatter(val_points[:, 0], val_points[:, 1], val_points[:, 2],
           c='orange', marker='^', s=40, label='Validation measurements')
ax.set_xlabel('Conductivity x')
ax.set_ylabel('Conductivity y')
ax.set_zlabel('Conductivity z')
ax.set_title('3D Effective Conductivity — Rotation Grid Sampling')
ax.legend()
plt.show()


# --- ConvexHull surface ---
hull = ConvexHull(all_points)
faces = hull.simplices
pv_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()
surface = pv.PolyData(all_points, pv_faces)

plotter_hull = pv.Plotter()
plotter_hull.add_mesh(surface, color='cyan', opacity=0.5, show_edges=True,
                      label='Convex Hull Surface')
plotter_hull.add_points(fit_points, color='red', point_size=10,
                        label='Fit measurements')
plotter_hull.add_points(val_points, color='orange', point_size=12,
                        label='Validation measurements')
plotter_hull.add_legend()
plotter_hull.add_axes()
plotter_hull.show()


# --- Fitted ellipsoid ---
sphere = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
sphere_pts = np.array(sphere.points)
ellipsoid_pts = (eigvecs @ np.diag(eigvals) @ sphere_pts.T).T
ellipsoid = sphere.copy()
ellipsoid.points = ellipsoid_pts

plotter = pv.Plotter()
plotter.add_mesh(ellipsoid, color='lightblue', opacity=0.4, label='Fitted Ellipsoid')
plotter.add_points(fit_points, color='red', point_size=8, label='Fit measurements')
plotter.add_points(val_points, color='orange', point_size=12, label='Validation measurements')

labels = ['k1 (max)', 'k2 (mid)', 'k3 (min)']
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

plotter.add_axes()
plotter.show()


# --- Triangulated shell surface of measured ellipsoid ---
from pathlib import Path

def build_continuous_shell_mesh(points_3d):
    """Build a triangulated shell mesh from measurement points.

    Normalises every point to the unit sphere, computes the convex hull
    of the unit directions (so every point appears in at least one triangle),
    then applies the face indices back to the original 3-D coordinates.
    """
    points = np.asarray(points_3d, dtype=float)
    if len(points) < 4:
        return None, None, None

    norms = np.linalg.norm(points, axis=1, keepdims=True)
    unit = points / np.where(norms > 0, norms, 1.0)

    hull = ConvexHull(unit)
    faces = hull.simplices

    pv_faces = np.hstack([
        np.full((len(faces), 1), 3, dtype=np.intp),
        faces,
    ]).ravel()

    mesh = pv.PolyData(points, pv_faces)
    point_ids = np.arange(1, len(points) + 1, dtype=int)
    return points, mesh, point_ids


# Build the mesh from all measurement points
shell_pts, shell_mesh, shell_ids = build_continuous_shell_mesh(all_points)

if shell_mesh is not None:
    point_cloud = pv.PolyData(shell_pts)
    point_cloud["point_id"] = shell_ids

    plotter_shell = pv.Plotter()
    plotter_shell.add_mesh(shell_mesh, color="lightsteelblue",
                           opacity=0.30, show_edges=True)
    plotter_shell.add_points(point_cloud, color="darkred", point_size=10,
                             label="Measurement points")
    plotter_shell.add_scalar_bar(title="Point ID")
    plotter_shell.add_axes()
    plotter_shell.add_title(
        f"Measured Ellipsoid — Triangulated Shell ({len(shell_ids)} points)")
    plotter_shell.camera_position = "iso"
    plotter_shell.show()


# --- Directional conductivity surface k(n) — no tensor assumption ---
# Interpolate k(n) directly from measurements onto a dense Fibonacci sphere.
# Each point on the sphere is scaled by k_interp(n), so the surface radius
# in direction n equals the measured effective conductivity.  No tensor model
# is imposed — this is the physically honest diagnostic representation.
from scipy.interpolate import RBFInterpolator

# Combine fit + validation measurements
all_meas_dirs = np.vstack([fit_directions, val_directions])   # (N, 3) unit vectors
all_meas_k    = np.concatenate([fit_k, val_k])                # (N,) conductivity values

# Also add antipodal copies: k(n) = k(-n) for a symmetric tensor.
# This prevents artefacts on the back of the sphere where data is sparse.
meas_dirs_sym = np.vstack([all_meas_dirs, -all_meas_dirs])
meas_k_sym    = np.concatenate([all_meas_k, all_meas_k])

# Dense Fibonacci lattice on the unit sphere (upper hemisphere + mirror)
n_fib    = 2000
golden   = (1.0 + np.sqrt(5.0)) / 2.0
ii       = np.arange(n_fib)
theta_f  = np.arccos(1.0 - 2.0 * (ii + 0.5) / n_fib)
phi_f    = 2.0 * np.pi * ii / golden
fib_dirs = np.column_stack([
    np.sin(theta_f) * np.cos(phi_f),
    np.sin(theta_f) * np.sin(phi_f),
    np.cos(theta_f),
])

# RBF interpolation in 3-D unit-vector space (thin-plate spline kernel)
rbf       = RBFInterpolator(meas_dirs_sym, meas_k_sym, kernel='thin_plate_spline', smoothing=0.1)
k_interp  = rbf(fib_dirs)
k_interp  = np.maximum(k_interp, 0.0)   # conductivity must be non-negative

# Scale each unit-sphere point by the interpolated k → 3D surface
surf_pts = fib_dirs * k_interp[:, np.newaxis]

# Triangulate via convex hull of unit directions
unit_norms = np.linalg.norm(surf_pts, axis=1, keepdims=True)
unit_dirs  = surf_pts / np.where(unit_norms > 0, unit_norms, 1.0)
hull_k     = ConvexHull(unit_dirs)
ksurf_faces = np.hstack([
    np.full((len(hull_k.simplices), 1), 3, dtype=np.intp),
    hull_k.simplices,
]).ravel()

ksurf_mesh = pv.PolyData(surf_pts, ksurf_faces)
ksurf_mesh["k(n)"] = k_interp

# Measurement points overlay
meas_cloud = pv.PolyData(all_meas_dirs * all_meas_k[:, np.newaxis])
meas_cloud["k_meas"] = all_meas_k

plotter_ksurf = pv.Plotter()
plotter_ksurf.add_mesh(
    ksurf_mesh,
    scalars="k(n)",
    cmap="plasma",
    opacity=0.75,
    show_edges=False,
    smooth_shading=True,
)
plotter_ksurf.add_points(
    meas_cloud,
    color="white",
    point_size=8,
    render_points_as_spheres=True,
    label="Measurements",
)
plotter_ksurf.add_scalar_bar(title="k(n)  [m/s]")
plotter_ksurf.add_axes()
plotter_ksurf.add_title("Directional Conductivity Surface k(n) — No Tensor Assumption")
plotter_ksurf.camera_position = "iso"
plotter_ksurf.show()
