import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Rotation utilities
# ============================================================
# Building rotation matricies for Z–X–Y convention


def rotation_matrix_zxy(z_deg, x_deg, y_deg):
    z, x, y = np.radians([z_deg, x_deg, y_deg])

    cz, sz = np.cos(z), np.sin(z)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)

    rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])

    ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])

    rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [0,   0,  1]])

    return rx @ ry @ rz


# ============================================================
# Box faces (local normals) [z,x,y]
# ============================================================
FACE_AXES = {
    "+x": np.array([1,  0,  0]),
    "-x": np.array([-1,  0,  0]),
    "+y": np.array([0,  1,  0]),
    "-y": np.array([0, -1,  0]),
    "+z": np.array([0,  0,  1]),
    "-z": np.array([0,  0, -1]),
}

# Colour of every face, used for plotting
FACE_COLORS = {
    "+x": "#eb0a1c",
    "-x": "#d00000",
    "+y": "#457b9d",
    "-y": "#1d3557",
    "+z": "#2a9d8f",
    "-z": "#1b7f6b",
}

# ============================================================
# Test mode: isotropic vs anisotropic (giving random noise)
# ============================================================
ISOTROPIC_TEST = True
ANISOTROPY_RANDOM_SCALE = 0.20
RANDOM_SEED = 42

# Optional integration hook for external DFN flow results.
# Keep False to preserve current mock behavior.
USE_DFN_HOOK = False
DARCY_GRADIENT_MAG = 1.0


# ============================================================
# Mock DFN – replaced by flow hook in real integration
# ============================================================
def mock_dfn_isotropic(active_faces):
    """
    Gives k-values based on measured flow q = -K grad(h)
    On each side
    Isotropic test case: same k for all faces, no randomness.
    """
    k = 1.5
    return {face: k for face in active_faces}


def mock_dfn_anisotropic_random(active_faces, rotation_matrix, rng):
    """Directional anisotropy + mild randomness"""
    k0 = 1.5
    results = {}

    for face in active_faces:
        n_local = FACE_AXES[face]
        n_global = rotation_matrix @ n_local

        # Use spherical direction angles to vary conductivity by orientation.
        # phi: azimuth in xy, theta: elevation from xy.
        phi = np.arctan2(n_global[1], n_global[0])
        theta = np.arctan2(n_global[2], np.linalg.norm(n_global[:2]) + 1e-12)

        directional = 1.0 + 0.18 * \
            np.cos(2.0 * phi) + 0.12 * np.sin(2.0 * theta)
        noise = 1.0 - rng.random() * ANISOTROPY_RANDOM_SCALE
        results[face] = k0 * directional * noise

    return results


# Defines which function to use, if its isotropic or anisotripic with random noise.
def mock_dfn_values(active_faces, rotation_matrix, rng):
    if ISOTROPIC_TEST:
        return mock_dfn_isotropic(active_faces)
    return mock_dfn_anisotropic_random(active_faces, rotation_matrix, rng)


# Integration hook placeholder: replace with real DFN flow data retrieval.
def dfn_flow_hook(active_faces, rotation_matrix, rot_label):
    """Hook for real DFN integration: return measured flow q per face.

    Expected return format:
        {
            "+x": q_plus_x,
            "-x": q_minus_x,
            ...
        }

    Notes:
    - q should be scalar flow aligned with chosen face direction convention.
    - active_faces provides which faces are required for this rotation step.
    - rotation_matrix and rot_label are included for traceability/mapping.
    """
    raise NotImplementedError(
        "Implement dfn_flow_hook(...) to return q per face from external DFN output."
    )


def k_from_darcy_q(q_by_face, gradient_mag, active_faces):
    """Convert DFN flow q to directional K using Darcy relation q = K * grad."""
    grad_abs = max(abs(float(gradient_mag)), 1e-12)
    return {face: abs(float(q_by_face[face])) / grad_abs for face in active_faces}


# ============================================================
# Rotation setup: base + repeted tilt in xz-plan + rotation in xy-plan
# ============================================================

tilt_y_angles = list(range(15, 90, 15))  # 15, 30, ..., 75
z_angles = list(range(15, 90, 15))  # 15, 30, ..., 75
z_angles_no_tilt = list(range(15, 90, 15))  # 15, 30, ..., 75

points = []
dirs = []
colors = []
rot_ids = []
face_ids = []

FACES_ALL = ["+x", "-x", "+y", "-y", "+z", "-z"]
FACES_XY = ["+x", "-x", "+y", "-y"]
FACES_XZ = ["+x", "-x", "+z", "-z"]
FACES_YZ = ["+y", "-y", "+z", "-z"]

# Plotinställningar för legend.
SHOW_LEGEND = False
LEGEND_LOC = "upper left"

ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv_files"
POINTS_CSV_PATH = CSV_DIR / "ny_rotation_points.csv"
TENSOR_CSV_PATH = CSV_DIR / "ny_rotation_tensor_for_continuum.csv"

RNG = np.random.default_rng(RANDOM_SEED)


# ============================================================
# Hjälpfunktion: calculate and add measurements for a given rotation and active faces.
# ============================================================

def add_measurements(R, active_faces, rot_label):
    if USE_DFN_HOOK:
        q_by_face = dfn_flow_hook(active_faces, R, rot_label)
        dfn_results = k_from_darcy_q(
            q_by_face, DARCY_GRADIENT_MAG, active_faces)
    else:
        dfn_results = mock_dfn_values(active_faces, R, RNG)

    for face in active_faces:
        n_local = FACE_AXES[face]
        n_global = R @ n_local

        flow_dir = -n_global
        k_val = dfn_results[face]

        points.append(k_val * flow_dir)
        dirs.append(flow_dir)
        rot_ids.append(rot_label)
        face_ids.append(face)

        # Face-based on colour, to easier see that all faces are part of the calculations.
        colors.append(FACE_COLORS[face])


# ============================================================
# Huvudsteg:
# 1) Startläge: spara ±x, ±y, ±z (ger 6 värden)
# 2) Rotation utan tilt: rotera bara i xy-planet och spara ±x, ±y.
# 3) För varje tilt i xz-planet (rotation runt y): spara ±x, ±z
# 4) Från varje tiltat läge: rotera i xy-planet runt z till 75°: spara ±x, ±y
# 5) Ny tredje sekvens från normalläge:
#    - Tilt 15° i yz-led (rotation runt x): spara ±y, ±z
#    - Rotera vidare i xy-led till 75° (rotation runt z), behåll yz-tilt: spara ±y, ±z
# ============================================================
# 1) Startläge: spara ±x, ±y, ±z (ger 6 värden) np.eye ger en rotationsmatris fylld med 0 runtom och 1 på diagonalen, dvs ingen rotation.
R_start = np.eye(3)
add_measurements(
    R_start,
    FACES_ALL,
    "start",
)

# 2) Rotation utan tilt: rotera bara i xy-planet och spara ±x, ±y.
for z_deg_no_tilt in z_angles_no_tilt:
    R_xy_no_tilt = rotation_matrix_zxy(z_deg_no_tilt, 0, 0)
    add_measurements(
        R_xy_no_tilt,
        FACES_XY,
        f"base_z{z_deg_no_tilt}",
    )

# 3) För varje tilt i xz-planet (rotation runt y): spara ±x, ±z
for tilt_y_deg in tilt_y_angles:
    R_tilt = rotation_matrix_zxy(0, 0, tilt_y_deg)
    add_measurements(
        R_tilt,
        FACES_XZ,
        f"tilt_y{tilt_y_deg}",
    )

    for z_deg in z_angles:
        R_xy = rotation_matrix_zxy(z_deg, 0, 0)
        R_combined = R_xy @ R_tilt
        add_measurements(
            R_combined,
            FACES_ALL,
            f"tilt_y{tilt_y_deg}_z{z_deg}",
        )


def add_yz_then_xy_sequence(step_name, tilt_x_deg, z_values):
    """Run one yz-tilt sequence followed by xy-rotations at fixed z angles."""
    # Step one, create tilt in yz plane
    r_yz = rotation_matrix_zxy(0, tilt_x_deg, 0)
    add_measurements(r_yz, FACES_YZ, f"{step_name}_yz_x{tilt_x_deg}")

    # Step 2: Rotate in xy-plane around z-axis, keeping the yz tilt, and save ±y, ±z for each rotation.
    for z_deg in z_values:
        r_xy = rotation_matrix_zxy(z_deg, 0, 0)
        r_combined = r_xy @ r_yz
        add_measurements(
            r_combined,
            FACES_YZ,
            f"{step_name}_yz_x{tilt_x_deg}_xy_z{z_deg}",
        )


# Tredje steg: ny sekvens som startar om från normalläge och kör yz = 15, 30, 45, 60, 75.
tilt_x_angles_step3 = list(range(15, 90, 15))  # 15, 30, ..., 75
z_angles_step3 = list(range(15, 90, 15))  # 15, 30, ..., 75

for tilt_x_deg in tilt_x_angles_step3:
    add_yz_then_xy_sequence("step3", tilt_x_deg, z_angles_step3)

# ============================================================
# Store data
# ============================================================
points = np.array(points)
dirs = np.array(dirs)
k_meas = np.linalg.norm(points, axis=1)
rot_ids = np.array(rot_ids)
colors = np.array(colors)


# ============================================================
# Tensor-fit: k(n) = nᵀ K n
# ============================================================
def fit_tensor_least_squares(direction_vectors, k_values):
    """Fit symmetric conductivity tensor K from directional measurements.

    Model: k = n^T K n
    where n is a unit direction vector.
    """
    nx, ny, nz = (
        direction_vectors[:, 0],
        direction_vectors[:, 1],
        direction_vectors[:, 2],
    )

    design = np.column_stack([
        nx**2,
        2 * nx * ny,
        2 * nx * nz,
        ny**2,
        2 * ny * nz,
        nz**2,
    ])

    coef, *_ = np.linalg.lstsq(design, k_values, rcond=None)

    k_tensor = np.array([
        [coef[0], coef[1], coef[2]],
        [coef[1], coef[3], coef[4]],
        [coef[2], coef[4], coef[5]],
    ])

    k_pred = design @ coef
    rmse = float(np.sqrt(np.mean((k_values - k_pred) ** 2)))
    return k_tensor, design, rmse


K_fit, A, rmse_fit = fit_tensor_least_squares(dirs, k_meas)

print("Rank(A):", np.linalg.matrix_rank(A))
print(
    "Test mode:",
    "isotropic" if ISOTROPIC_TEST else f"anisotropic_random (seed={RANDOM_SEED}, scale={ANISOTROPY_RANDOM_SCALE})",
)
# Dubbelkolla ifall nått tal inte gick att lösa
expected_points = 6 + 4 * len(z_angles_no_tilt) + len(tilt_y_angles) * \
    (4 + 6 * len(z_angles)) + len(tilt_x_angles_step3) * \
    (4 + 4 * len(z_angles_step3))
print("Total points:", len(points), f"(expected {expected_points})")

rot_order = list(dict.fromkeys(rot_ids.tolist()))
for rot_label in rot_order:
    mask = rot_ids == rot_label
    unique_faces = sorted(set(face_ids[i] for i in np.where(mask)[0]))
    print(f"{rot_label}: {np.sum(mask)} pkt, faces={unique_faces}")

print(f"RMSE residual (least squares): {rmse_fit:.6e} (0 = perfect ellipsoid)")


def build_ellipsoid_mesh_from_tensor(k_tensor, n_u=60, n_v=30):
    """Create ellipsoid mesh from eigen decomposition of fitted tensor."""
    evals, evecs = np.linalg.eigh(k_tensor)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 1e-12)
    evecs = evecs[:, order]

    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    ellipsoid = (evecs @ np.diag(evals) @ sphere.T).T

    ex = ellipsoid[:, 0].reshape(x.shape)
    ey = ellipsoid[:, 1].reshape(y.shape)
    ez = ellipsoid[:, 2].reshape(z.shape)
    return ex, ey, ez, evals, evecs


def export_rotation_csvs():
    """Export point-wise data and tensor data to csv_files/."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Point-wise export: each measured point with global direction.
    # kx,ky,kz are the plotted K-vector components; k_value is its magnitude.
    header_points = (
        "point_id,rotation_label,face,kx,ky,kz,"
        "k_value,dir_x,dir_y,dir_z\n"
    )
    with open(POINTS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(header_points)
        for i, (p, d, rot_label, face) in enumerate(zip(points, dirs, rot_ids, face_ids), start=1):
            k_value = float(np.linalg.norm(p))
            f.write(
                f"{i},{rot_label},{face},"
                f"{p[0]:.10g},{p[1]:.10g},{p[2]:.10g},"
                f"{k_value:.10g},{d[0]:.10g},{d[1]:.10g},{d[2]:.10g}\n"
            )

    # Tensor export in the exact column format expected by f_continuum_model.py.
    # Single row at angle 0.0 representing fitted global tensor.
    header_tensor = (
        "angle_deg,k_xx,k_xy,k_xz,k_yx,k_yy,k_yz,k_zx,k_zy,k_zz\n"
    )
    with open(TENSOR_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(header_tensor)
        f.write(
            "0.0,"
            f"{K_fit[0, 0]:.10g},{K_fit[0, 1]:.10g},{K_fit[0, 2]:.10g},"
            f"{K_fit[1, 0]:.10g},{K_fit[1, 1]:.10g},{K_fit[1, 2]:.10g},"
            f"{K_fit[2, 0]:.10g},{K_fit[2, 1]:.10g},{K_fit[2, 2]:.10g}\n"
        )

    print(f"Saved point CSV: {POINTS_CSV_PATH}")
    print(f"Saved tensor CSV for continuum model: {TENSOR_CSV_PATH}")


export_rotation_csvs()


# ============================================================
# Plot
# ============================================================
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

markers = ["o", "^", "s", "D", "P", "X"]

# Återanvänd överlappande ±y-värden i visualiseringen: plotta bara första unika.
plot_mask = np.ones(len(points), dtype=bool)
seen_y_keys = set()
for i, (p, face, rot_label) in enumerate(zip(points, face_ids, rot_ids)):
    keep_initial_xy = str(rot_label) == "start" or str(
        rot_label).startswith("base_z")
    if face in ["+y", "-y"] and not keep_initial_xy:
        y_key = tuple(np.round(p, 8))
        if y_key in seen_y_keys:
            plot_mask[i] = False
        else:
            seen_y_keys.add(y_key)

points_plot = points[plot_mask]
rot_ids_plot = rot_ids[plot_mask]
face_ids_plot = np.array(face_ids)[plot_mask]
colors_plot = colors[plot_mask]

# Visnings-offset: separerar överlappande punkter utan att ändra beräkningen.
points_plot_display = points_plot.copy()
for i, face in enumerate(face_ids_plot):
    points_plot_display[i] = points_plot_display[i] + 0.03 * FACE_AXES[face]

print("Plotted points after y-reuse:", len(points_plot))
if np.any(rot_ids == "tilt_y30_z30"):
    raw_3030 = [face_ids[i] for i in np.where(rot_ids == "tilt_y30_z30")[0]]
    plot_3030 = [face_ids_plot[i]
                 for i in np.where(rot_ids_plot == "tilt_y30_z30")[0]]
    print("tilt_y30_z30 raw faces:", sorted(raw_3030))
    print("tilt_y30_z30 plotted faces:", sorted(plot_3030))

for i, rot_label in enumerate(rot_order):
    mask = rot_ids_plot == rot_label
    if np.sum(mask) == 0:
        continue
    ax.scatter(
        points_plot_display[mask, 0],
        points_plot_display[mask, 1],
        points_plot_display[mask, 2],
        c=colors_plot[mask],
        s=60,
        alpha=0.95,
        marker=markers[i % len(markers)],
        edgecolors="black",
        linewidths=0.35,
        depthshade=False,
        label=f"{rot_label} ({np.sum(mask)} plot)",
    )

# Etiketter gör att överlappande punkter ändå kan identifieras.
for p, rot_label, face in zip(points_plot_display, rot_ids_plot, face_ids_plot):
    ax.text(
        p[0] + 0.03,
        p[1] + 0.03,
        p[2] + 0.03,
        f"{rot_label}:{face}",
        fontsize=7,
        alpha=0.85,
    )

lim = np.max(np.abs(points)) * 1.2
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_box_aspect([1, 1, 1])

ax.set_xlabel("Kx")
ax.set_ylabel("Ky")
ax.set_zlabel("Kz")
ax.set_title(
    "Directional conductivity from flow experiments\n"
    "(start + repeated tilt in xz + rotation in xy + extra yz->xy step)"
)
if SHOW_LEGEND:
    legend = ax.legend(loc=LEGEND_LOC, fontsize=8)
    legend.set_draggable(True)

plt.tight_layout()
plt.show()


# ============================================================
# Ellipsoid fit plot (least squares)
# ============================================================
fig_fit = plt.figure(figsize=(8, 7))
ax_fit = fig_fit.add_subplot(111, projection="3d")

ex, ey, ez, evals_fit, evecs_fit = build_ellipsoid_mesh_from_tensor(K_fit)

ax_fit.plot_surface(ex, ey, ez, color="#8ecae6", alpha=0.28, linewidth=0)
ax_fit.scatter(points[:, 0], points[:, 1],
               points[:, 2], c=colors, s=16, alpha=0.6)

# Principal axes from fitted tensor
axis_colors = ["#d62828", "#2a9d8f", "#1d3557"]
for i in range(3):
    vec = evecs_fit[:, i] * evals_fit[i]
    ax_fit.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                  color=axis_colors[i], linewidth=2)

lim_fit = max(np.max(np.abs(points)) * 1.2,
              np.max(np.abs(np.array([ex, ey, ez]))))
ax_fit.set_xlim(-lim_fit, lim_fit)
ax_fit.set_ylim(-lim_fit, lim_fit)
ax_fit.set_zlim(-lim_fit, lim_fit)
ax_fit.set_box_aspect([1, 1, 1])
ax_fit.set_xlabel("Kx")
ax_fit.set_ylabel("Ky")
ax_fit.set_zlabel("Kz")
ax_fit.set_title("Least-squares fitted conductivity ellipsoid")

plt.tight_layout()
plt.show()


# ============================================================
# Summary
# ============================================================
eigvals, eigvecs = np.linalg.eigh(K_fit)
order = np.argsort(eigvals)[::-1]

print("\nFitted conductivity tensor K:")
print(K_fit)

print("\nPrincipal conductivities:")
print(eigvals[order])

print("\nPrincipal directions (columns):")
print(eigvecs[:, order])
