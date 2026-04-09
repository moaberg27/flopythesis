import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Rotation utilities
# ============================================================
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
                   [sz, cz, 0],
                   [0, 0, 1]])

    return rx @ ry @ rz


def directional_k(K, n):
    n = np.asarray(n)
    n /= np.linalg.norm(n)
    return float(n.T @ K @ n)


# ============================================================
# Angle setup
# ============================================================
def build_angles_to_90(step):
    if 90 % step != 0:
        raise ValueError("step must divide 90")
    return list(range(0, 90, step))


step_angle = 15
angles_deg = build_angles_to_90(step_angle)

# ============================================================
# Direction helper
# ============================================================
def build_directions(axes):
    d = {"+x": False, "-x": False,
         "+y": False, "-y": False,
         "+z": False, "-z": False}
    for a in axes:
        d[f"+{a}"] = True
        d[f"-{a}"] = True
    return d


# ============================================================
# Rotation plan (DIN LOGIK)
# ============================================================
rotation_plan = []

for y_deg in angles_deg:
    for x_deg in angles_deg:
        for z_deg in angles_deg:

            # Basfall
            if x_deg == 0 and y_deg == 0 and z_deg == 0:
                directions = build_directions(["x", "y", "z"])

            # Ren z-rotation
            elif x_deg == 0 and y_deg == 0:
                directions = build_directions(["x", "y"])

            # x-rotation
            elif x_deg != 0 and y_deg == 0:
                directions = build_directions(["x", "z"])

            # y-rotation
            elif y_deg != 0 and x_deg == 0:
                directions = build_directions(["y", "z"])

            # x + y rotation
            else:
                directions = build_directions(["y", "z"])

            rotation_plan.append(((z_deg, x_deg, y_deg), directions))


# ============================================================
# Mock DFN (isotropic test)
# ============================================================
def mock_dfn_isotropic(rotation, directions):
    k = 1.5
    return {
        d: k if active else None
        for d, active in directions.items()
    }


# ============================================================
# Run DFN
# ============================================================
dfn_results = {}

for rot, directions in rotation_plan:
    dfn_results[rot] = mock_dfn_isotropic(rot, directions)


# ============================================================
# Collect measurement points
# ============================================================
points = []
dirs = []

for rot, directions in rotation_plan:
    r = rotation_matrix_zxy(*rot)
    ex, ey, ez = r[:, 0], r[:, 1], r[:, 2]
    res = dfn_results[rot]

    if res["+x"] is not None:
        points.append(res["+x"] * ex)
        dirs.append(ex)
    if res["-x"] is not None:
        points.append(res["-x"] * -ex)
        dirs.append(-ex)

    if res["+y"] is not None:
        points.append(res["+y"] * ey)
        dirs.append(ey)
    if res["-y"] is not None:
        points.append(res["-y"] * -ey)
        dirs.append(-ey)

    if res["+z"] is not None:
        points.append(res["+z"] * ez)
        dirs.append(ez)
    if res["-z"] is not None:
        points.append(res["-z"] * -ez)
        dirs.append(-ez)

points = np.array(points)
dirs = np.array(dirs)
k_meas = np.linalg.norm(points, axis=1)

# ============================================================
# Tensor fit
# ============================================================
nx, ny, nz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

A = np.column_stack([
    nx**2,
    2*nx*ny,
    2*nx*nz,
    ny**2,
    2*ny*nz,
    nz**2
])

print("Rank(A):", np.linalg.matrix_rank(A))

coef, *_ = np.linalg.lstsq(A, k_meas, rcond=None)

K_fit = np.array([
    [coef[0], coef[1], coef[2]],
    [coef[1], coef[3], coef[4]],
    [coef[2], coef[4], coef[5]],
])

eigvals, eigvecs = np.linalg.eigh(K_fit)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]

# ============================================================
# Plot: ALL POINTS
# ============================================================
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

order_color = np.arange(len(points))

sc = ax.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    c=order_color,
    cmap="turbo",
    s=28,
    alpha=0.9
)

lim = np.max(np.abs(points)) * 1.15
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_box_aspect([1, 1, 1])

ax.set_xlabel("Kx")
ax.set_ylabel("Ky")
ax.set_zlabel("Kz")
ax.set_title("Directional Hydraulic Conductivity – All Measurements")

plt.colorbar(sc, ax=ax, label="Measurement index")
plt.tight_layout()
plt.show()

# ============================================================
# Summary
# ============================================================
print("\nFitted conductivity tensor K:")
print(K_fit)
print("Principal values:", eigvals)
