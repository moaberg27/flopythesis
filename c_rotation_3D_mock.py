"""Continuum assumption test in 3D using a mock DFN.

Concept mirrors the 2D workflow:
1) Run mock DFN for several model rotations.
2) Collect directional conductivity measurements.
3) Fit a symmetric 3D conductivity tensor K.
4) Evaluate fit quality (RMSE) and visualize measurement cloud + fitted ellipsoid.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv_files"
TENSOR_EXPORT_PATH = CSV_DIR / "tensor_sim_one.csv"


def rotation_matrix_zxy(z_deg, x_deg, y_deg):
    """Return local-to-global rotation for sequence z -> x -> y.

    Input tuple convention in this script is (z_deg, x_deg, y_deg).
    """
    z_rad = np.radians(z_deg)
    x_rad = np.radians(x_deg)
    y_rad = np.radians(y_deg)

    cz, sz = np.cos(z_rad), np.sin(z_rad)
    cx, sx = np.cos(x_rad), np.sin(x_rad)
    cy, sy = np.cos(y_rad), np.sin(y_rad)

    rz = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ])
    ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ])
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ])
    return ry @ rx @ rz


def directional_k(k_tensor, direction_vec):
    """Directional conductivity k(n) = n^T K n for unit direction n."""
    n = np.asarray(direction_vec, dtype=float)
    n = n / np.linalg.norm(n)
    return float(n.T @ k_tensor @ n)


# Rotation set in 3D: (z, x, y) in degrees
# Order follows your request:
# 1) z from 0..75 at fixed x,y
# 2) then next x-step and sweep z again
# 3) then next y-step and repeat
angles_deg = [0, 15, 30, 45, 60, 75]
rotations_3d = [
    (z_deg, x_deg, y_deg)
    for y_deg in angles_deg
    for x_deg in angles_deg
    for z_deg in angles_deg
]
print(
    f"Total rotations: {len(rotations_3d)} (expected {len(angles_deg) ** 3})")


# DFN results collector
dfn_results = {}


def add_rotation(rotation_deg, k_x_pos, k_x_neg, k_y_pos, k_y_neg, k_z_pos, k_z_neg):
    """Store 6 K values from one 3D rotation."""
    dfn_results[rotation_deg] = (
        k_x_pos, k_x_neg, k_y_pos, k_y_neg, k_z_pos, k_z_neg)
    print(
        f"Added rotation {rotation_deg}: "
        f"K(+x)={k_x_pos:.4f}, K(-x)={k_x_neg:.4f}, "
        f"K(+y)={k_y_pos:.4f}, K(-y)={k_y_neg:.4f}, "
        f"K(+z)={k_z_pos:.4f}, K(-z)={k_z_neg:.4f}"
    )


# Underlying "true" anisotropic tensor used by mock DFN
np.random.seed(42)
true_principal = np.array([6.0, 2.0, 0.8])
r_true = rotation_matrix_zxy(z_deg=25.0, x_deg=35.0, y_deg=10.0)
k_true = r_true @ np.diag(true_principal) @ r_true.T


def mock_dfn(rotation_deg, noise=0.10):
    """Compute mock directional K for local +/-x, +/-y, +/-z at one 3D rotation."""
    r = rotation_matrix_zxy(*rotation_deg)

    # Local basis vectors expressed in global coordinates
    ex = r[:, 0]
    ey = r[:, 1]
    ez = r[:, 2]

    def noisy_k(vec):
        k = directional_k(k_true, vec)
        k = k * (1.0 + noise * np.random.randn())
        return max(k, 1e-8)

    return (
        noisy_k(ex),
        noisy_k(-ex),
        noisy_k(ey),
        noisy_k(-ey),
        noisy_k(ez),
        noisy_k(-ez),
    )

# test with isotropic tensor (should give same K in all directions)
#def mock_dfn(rotation_deg):
 #   k_iso = 1.5
 #   return (3, 3, k_iso, 3, k_iso, 5)

# Run mock DFN for all rotations and store results
for rot in rotations_3d:
    add_rotation(rot, *mock_dfn(rot))

assert set(dfn_results.keys()) == set(
    rotations_3d), "Missing one or more 3D rotations."


# Map results to global 3D vectors for plotting/fitting
points_x_pos = []
points_x_neg = []
points_y_pos = []
points_y_neg = []
points_z_pos = []
points_z_neg = []

for rot in rotations_3d:
    k_x_pos, k_x_neg, k_y_pos, k_y_neg, k_z_pos, k_z_neg = dfn_results[rot]
    r = rotation_matrix_zxy(*rot)
    ex = r[:, 0]
    ey = r[:, 1]
    ez = r[:, 2]

    points_x_pos.append(k_x_pos * ex)
    points_x_neg.append(k_x_neg * (-ex))
    points_y_pos.append(k_y_pos * ey)
    points_y_neg.append(k_y_neg * (-ey))
    points_z_pos.append(k_z_pos * ez)
    points_z_neg.append(k_z_neg * (-ez))

    print(
        f"Rotation {rot}: "
        f"K(+x)={k_x_pos:.3f}, K(-x)={k_x_neg:.3f}, "
        f"K(+y)={k_y_pos:.3f}, K(-y)={k_y_neg:.3f}, "
        f"K(+z)={k_z_pos:.3f}, K(-z)={k_z_neg:.3f}"
    )

points_x_pos = np.array(points_x_pos)
points_x_neg = np.array(points_x_neg)
points_y_pos = np.array(points_y_pos)
points_y_neg = np.array(points_y_neg)
points_z_pos = np.array(points_z_pos)
points_z_neg = np.array(points_z_neg)

all_points = np.vstack(
    [points_x_pos, points_x_neg, points_y_pos,
        points_y_neg, points_z_pos, points_z_neg]
)


# Plot 1: raw directional measurements (3D)
fig1 = plt.figure(figsize=(8, 7))
ax1 = fig1.add_subplot(111, projection="3d")
ax1.scatter(*points_x_pos.T, s=45, marker="o", label="+x")
ax1.scatter(*points_x_neg.T, s=45, marker="o", label="-x")
ax1.scatter(*points_y_pos.T, s=45, marker="^", label="+y")
ax1.scatter(*points_y_neg.T, s=45, marker="^", label="-y")
ax1.scatter(*points_z_pos.T, s=45, marker="s", label="+z")
ax1.scatter(*points_z_neg.T, s=45, marker="s", label="-z")
ax1.set_xlabel("Kx")
ax1.set_ylabel("Ky")
ax1.set_zlabel("Kz")
ax1.set_title("Directional Hydraulic Conductivity (3D mock DFN)")
ax1.legend(loc="upper left")

lim = np.max(np.abs(all_points)) * 1.15
ax1.set_xlim(-lim, lim)
ax1.set_ylim(-lim, lim)
ax1.set_zlim(-lim, lim)
ax1.set_box_aspect([1, 1, 1])
plt.tight_layout()


# Fit symmetric 3D conductivity tensor via least squares
# k_i = n_i^T K n_i
# K = [[Kxx, Kxy, Kxz],
#      [Kxy, Kyy, Kyz],
#      [Kxz, Kyz, Kzz]]
dirs = []
k_meas = []

for rot in rotations_3d:
    k_x_pos, k_x_neg, k_y_pos, k_y_neg, k_z_pos, k_z_neg = dfn_results[rot]
    r = rotation_matrix_zxy(*rot)
    ex = r[:, 0]
    ey = r[:, 1]
    ez = r[:, 2]

    dirs.extend([ex, -ex, ey, -ey, ez, -ez])
    k_meas.extend([k_x_pos, k_x_neg, k_y_pos, k_y_neg, k_z_pos, k_z_neg])

dirs = np.array(dirs)
k_meas = np.array(k_meas)

nx = dirs[:, 0]
ny = dirs[:, 1]
nz = dirs[:, 2]

a = np.column_stack(
    [
        nx**2,
        2.0 * nx * ny,
        2.0 * nx * nz,
        ny**2,
        2.0 * ny * nz,
        nz**2,
    ]
)

result, _, _, _ = np.linalg.lstsq(a, k_meas, rcond=None)
kxx, kxy, kxz, kyy, kyz, kzz = result

k_fit = np.array(
    [
        [kxx, kxy, kxz],
        [kxy, kyy, kyz],
        [kxz, kyz, kzz],
    ]
)

# Principal values and axes
eigvals, eigvecs = np.linalg.eigh(k_fit)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

# Keep plotting stable if noise makes a very small negative eigenvalue
eigvals_plot = np.clip(eigvals, 1e-8, None)

# Fit quality
k_pred = a @ result
rmse = np.sqrt(np.mean((k_meas - k_pred) ** 2))
rel_rmse = rmse / np.mean(k_meas) * 100.0

print("\n" + "=" * 66)
print("  Fitted 3D Conductivity Tensor (for FLOPY upscaling)")
print(f"  K = [[{kxx:.4e}, {kxy:.4e}, {kxz:.4e}],")
print(f"       [{kxy:.4e}, {kyy:.4e}, {kyz:.4e}],")
print(f"       [{kxz:.4e}, {kyz:.4e}, {kzz:.4e}]]")
print(
    f"  Principal values: k1 = {eigvals[0]:.4e}, "
    f"k2 = {eigvals[1]:.4e}, k3 = {eigvals[2]:.4e}"
)
print(f"  Anisotropy ratio: {eigvals[0] / max(eigvals[2], 1e-12):.2f}")
print(f"  RMSE residual:    {rmse:.4e}  (relative: {rel_rmse:.2f}%)")
if rel_rmse < 5:
    print("  --> Good ellipsoid fit: continuum assumption likely valid")
elif rel_rmse < 15:
    print("  --> Moderate fit: continuum assumption approximate")
else:
    print("  --> Poor fit: continuum assumption may NOT hold")
print("=" * 66)


# Save final fitted tensor + orientation for use in continuum model
CSV_DIR.mkdir(parents=True, exist_ok=True)
tensor_export = np.array(
    [
        [
            0.0,        # angle_deg placeholder for compatibility in continuum import
            kxx,
            kxy,
            kxz,
            kxy,
            kyy,
            kyz,
            kxz,
            kyz,
            kzz,
            eigvals[0],
            eigvals[1],
            eigvals[2],
            eigvecs[0, 0],
            eigvecs[1, 0],
            eigvecs[2, 0],
            eigvecs[0, 1],
            eigvecs[1, 1],
            eigvecs[2, 1],
            eigvecs[0, 2],
            eigvecs[1, 2],
            eigvecs[2, 2],
            rmse,
            rel_rmse,
        ]
    ]
)

np.savetxt(
    TENSOR_EXPORT_PATH,
    tensor_export,
    delimiter=",",
    header=(
        "angle_deg,k_xx,k_xy,k_xz,k_yx,k_yy,k_yz,k_zx,k_zy,k_zz,"
        "k1,k2,k3,v1_x,v1_y,v1_z,v2_x,v2_y,v2_z,v3_x,v3_y,v3_z,"
        "rmse,rel_rmse_percent"
    ),
    comments="",
)
print(f"Saved fitted tensor and orientation to: {TENSOR_EXPORT_PATH}")


# Plot 2: measurements + fitted ellipsoid + principal axes
u = np.linspace(0.0, 2.0 * np.pi, 64)
v = np.linspace(0.0, np.pi, 32)
uu, vv = np.meshgrid(u, v)

# Unit sphere parametric points
sx = np.cos(uu) * np.sin(vv)
sy = np.sin(uu) * np.sin(vv)
sz = np.cos(vv)

# Build ellipsoid in principal basis and rotate to global basis
xp = eigvals_plot[0] * sx
yp = eigvals_plot[1] * sy
zp = eigvals_plot[2] * sz

ellipsoid = np.stack([xp.ravel(), yp.ravel(), zp.ravel()], axis=0)
ellipsoid_global = eigvecs @ ellipsoid
xe = ellipsoid_global[0].reshape(sx.shape)
ye = ellipsoid_global[1].reshape(sx.shape)
ze = ellipsoid_global[2].reshape(sz.shape)

fig2 = plt.figure(figsize=(9, 8))
ax2 = fig2.add_subplot(111, projection="3d")
ax2.scatter(*points_x_pos.T, s=35, marker="o", label="+x")
ax2.scatter(*points_x_neg.T, s=35, marker="o", label="-x")
ax2.scatter(*points_y_pos.T, s=35, marker="^", label="+y")
ax2.scatter(*points_y_neg.T, s=35, marker="^", label="-y")
ax2.scatter(*points_z_pos.T, s=35, marker="s", label="+z")
ax2.scatter(*points_z_neg.T, s=35, marker="s", label="-z")
ax2.plot_wireframe(xe, ye, ze, rstride=2, cstride=2, linewidth=0.6, alpha=0.5)

origin = np.zeros(3)
axis_colors = ["r", "g", "b"]
for i, color in enumerate(axis_colors):
    axis_vec = eigvals_plot[i] * eigvecs[:, i]
    ax2.quiver(
        origin[0],
        origin[1],
        origin[2],
        axis_vec[0],
        axis_vec[1],
        axis_vec[2],
        color=color,
        linewidth=2.0,
    )

ax2.text2D(
    0.02,
    0.98,
    (
        f"k1={eigvals[0]:.3f}, k2={eigvals[1]:.3f}, k3={eigvals[2]:.3f}\n"
        f"RMSE={rel_rmse:.2f}%"
    ),
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
)

ax2.set_xlabel("Kx")
ax2.set_ylabel("Ky")
ax2.set_zlabel("Kz")
ax2.set_title("Directional K with Fitted Conductivity Ellipsoid (3D)")
ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.set_zlim(-lim, lim)
ax2.set_box_aspect([1, 1, 1])
ax2.legend(loc="upper left")
plt.tight_layout()
plt.show()
