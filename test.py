import numpy as np
import matplotlib.pyplot as plt
 
# ============================================================
# Rotation utilities (same convention as your code)
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
 
 
# ============================================================
# 1. TRUE conductivity tensor (synthetic, unknown to inversion)
# ============================================================
test_isotropic = False
 
if test_isotropic:
    K_true = np.eye(3) * 1.0
else:
    K_true = np.array([
        [2.0, 0.4, 0.1],
        [0.4, 1.0, 0.0],
        [0.1, 0.0, 0.3]
    ])
 
K_true = 0.5 * (K_true + K_true.T)  # enforce symmetry
 
 
# ============================================================
# 2. Rotation plan (systematic directions, like packer tests)
# ============================================================
def build_angles_to_90(step):
    if 90 % step != 0:
        raise ValueError("step must divide 90")
    return list(range(0, 90, step))
 
 
def build_directions(axes):
    d = {"+x": False, "-x": False,
         "+y": False, "-y": False,
         "+z": False, "-z": False}
    for a in axes:
        d[f"+{a}"] = True
        d[f"-{a}"] = True
    return d
 
 
step_angle = 15
angles_deg = build_angles_to_90(step_angle)
 
rotation_plan = []
 
for y_deg in angles_deg:
    for x_deg in angles_deg:
        for z_deg in angles_deg:
 
            if x_deg == 0 and y_deg == 0 and z_deg == 0:
                directions = build_directions(["x", "y", "z"])
            elif x_deg == 0 and y_deg == 0:
                directions = build_directions(["x", "y"])
            elif x_deg != 0 and y_deg == 0:
                directions = build_directions(["x", "z"])
            elif y_deg != 0 and x_deg == 0:
                directions = build_directions(["y", "z"])
            else:
                directions = build_directions(["y", "z"])
 
            rotation_plan.append(((z_deg, x_deg, y_deg), directions))
 
 
# ============================================================
# 3. Synthetic "DFN replacement":
#    Darcy flux from rotated gradients
# ============================================================
dirs = []
k_meas = []
fluxes = []
 
for rot, directions in rotation_plan:
    R = rotation_matrix_zxy(*rot)
    ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
 
    for lab, n in {
        "+x":  ex, "-x": -ex,
        "+y":  ey, "-y": -ey,
        "+z":  ez, "-z": -ez
    }.items():
 
        if not directions[lab]:
            continue
 
        n = n / np.linalg.norm(n)
        q = -K_true @ n              # Darcy: q = -K ∇h
        k_dir = np.linalg.norm(q)   # scalar "measured" conductivity
 
        dirs.append(n)
        k_meas.append(k_dir)
        fluxes.append(q)
 
dirs = np.array(dirs)
k_meas = np.array(k_meas)
fluxes = np.array(fluxes)
 
 
# ============================================================
# 4. Least-squares tensor fit (Niemi eq. 10–11 analogue)
# ============================================================
nx, ny, nz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
 
A = np.column_stack([
    nx**2,
    2 * nx * ny,
    2 * nx * nz,
    ny**2,
    2 * ny * nz,
    nz**2
])
 
print("Rank(A):", np.linalg.matrix_rank(A))
 
coef, *_ = np.linalg.lstsq(A, k_meas, rcond=None)
 
K_fit = np.array([
    [coef[0], coef[1], coef[2]],
    [coef[1], coef[3], coef[4]],
    [coef[2], coef[4], coef[5]]
])
 
 
# ============================================================
# 5. Eigen-analysis (principal directions)
# ============================================================
eig_true, vec_true = np.linalg.eigh(K_true)
eig_fit, vec_fit = np.linalg.eigh(K_fit)
 
eig_true = eig_true[::-1]
eig_fit = eig_fit[::-1]
 
 
# ============================================================
# 6. Visualisation
# ============================================================
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
 
k_points = dirs * k_meas[:, None]
 
ax.scatter(
    k_points[:, 0],
    k_points[:, 1],
    k_points[:, 2],
    c=k_meas,
    cmap="turbo",
    s=28
)
 
lim = np.max(np.abs(k_points)) * 1.2
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_box_aspect([1, 1, 1])
 
ax.set_xlabel("qx")
ax.set_ylabel("qy")
ax.set_zlabel("qz")
ax.set_title("Directional conductivity values (K mapped onto directions)")
 
plt.tight_layout()
plt.show()
 
 
# ============================================================
# 7. Summary
# ============================================================
print("\nTRUE tensor:")
print(K_true)