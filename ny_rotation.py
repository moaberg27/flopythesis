import numpy as np
import matplotlib.pyplot as plt

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
                   [sz,  cz, 0],
                   [0,   0,  1]])

    return rx @ ry @ rz


# ============================================================
# Box faces (lokala normaler)
# ============================================================
FACE_AXES = {
    "+x": np.array([1,  0,  0]),
    "-x": np.array([-1,  0,  0]),
    "+y": np.array([0,  1,  0]),
    "-y": np.array([0, -1,  0]),
    "+z": np.array([0,  0,  1]),
    "-z": np.array([0,  0, -1]),
}

FACE_COLORS = {
    "+x": "#e63946",
    "-x": "#d00000",
    "+y": "#457b9d",
    "-y": "#1d3557",
    "+z": "#2a9d8f",
    "-z": "#1b7f6b",
}


# ============================================================
# Mock DFN – ersätts av riktig flödesberäkning
# ============================================================
def mock_dfn_isotropic(active_faces):
    """
    Returnerar k-värden baserat på uppmätt flöde q = -K grad(h)
    Här isotropt testfall.
    """
    k = 1.5
    return {face: k for face in active_faces}


# ============================================================
# Rotation setup: bas + upprepad tilt i xz-plan + rotation i xy-plan
# ============================================================
tilt_y_angles = list(range(15, 90, 15))  # 15, 30, ..., 75
z_angles = list(range(15, 90, 15))  # 15, 30, ..., 75

points = []
dirs = []
colors = []
rot_ids = []
face_ids = []

# Plotinställningar för legend.
SHOW_LEGEND = False
LEGEND_LOC = "upper left"


# ============================================================
# Hjälpfunktion: beräkna och lagra mätpunkter
# ============================================================
def add_measurements(R, active_faces, rot_label):
    dfn_results = mock_dfn_isotropic(active_faces)

    for face in active_faces:
        n_local = FACE_AXES[face]
        n_global = R @ n_local

        # ====================================================
        # VIKTIGT:
        # Konduktiviteten hör till FLÖDESRIKTNINGEN
        # q = -K grad(h) ⇒ flöde går MOT facen
        # ====================================================
        flow_dir = -n_global
        k_val = dfn_results[face]

        points.append(k_val * flow_dir)
        dirs.append(flow_dir)
        rot_ids.append(rot_label)
        face_ids.append(face)

        # Face-baserad färg gör det lättare att se att alla faces finns med.
        colors.append(FACE_COLORS[face])


# ============================================================
# Huvudsteg:
# 1) Startläge: spara ±x, ±y, ±z
# 2) För varje tilt i xz-planet (rotation runt y): spara ±x, ±z
# 3) Från varje tiltat läge: rotera i xy-planet runt z till 75°: spara ±x, ±y, ±z
# 4) Ny tredje sekvens från normalläge:
#    - Tilt 15° i yz-led (rotation runt x): spara ±y, ±z
#    - Rotera vidare i xy-led till 75° (rotation runt z), behåll yz-tilt: spara ±y, ±z
# ============================================================
R_start = np.eye(3)
add_measurements(
    R_start,
    ["+x", "-x", "+y", "-y", "+z", "-z"],
    "start",
)

for tilt_y_deg in tilt_y_angles:
    R_tilt = rotation_matrix_zxy(0, 0, tilt_y_deg)
    add_measurements(
        R_tilt,
        ["+x", "-x", "+z", "-z"],
        f"tilt_y{tilt_y_deg}",
    )

    for z_deg in z_angles:
        R_xy = rotation_matrix_zxy(z_deg, 0, 0)
        R_combined = R_xy @ R_tilt
        add_measurements(
            R_combined,
            ["+x", "-x", "+y", "-y", "+z", "-z"],
            f"tilt_y{tilt_y_deg}_z{z_deg}",
        )

# Tredje steg: ny sekvens som startar om från normalläge.
tilt_x_deg_step3 = 15
z_angles_step3 = list(range(15, 90, 15))  # 15, 30, ..., 75

R_step3_yz = rotation_matrix_zxy(0, tilt_x_deg_step3, 0)
add_measurements(
    R_step3_yz,
    ["+y", "-y", "+z", "-z"],
    f"step3_yz_x{tilt_x_deg_step3}",
)

for z_deg_step3 in z_angles_step3:
    R_step3_xy = rotation_matrix_zxy(z_deg_step3, 0, 0)
    R_step3_combined = R_step3_xy @ R_step3_yz
    add_measurements(
        R_step3_combined,
        ["+y", "-y", "+z", "-z"],
        f"step3_yz_x{tilt_x_deg_step3}_xy_z{z_deg_step3}",
    )

# Fjärde steg: upprepa samma sekvens för yz = 30, 45, 60, 75.
tilt_x_angles_step4 = [30, 45, 60, 75]
z_angles_step4 = [15, 30, 45, 50, 75]

for tilt_x_deg_step4 in tilt_x_angles_step4:
    R_step4_yz = rotation_matrix_zxy(0, tilt_x_deg_step4, 0)
    add_measurements(
        R_step4_yz,
        ["+y", "-y", "+z", "-z"],
        f"step4_yz_x{tilt_x_deg_step4}",
    )

    for z_deg_step4 in z_angles_step4:
        R_step4_xy = rotation_matrix_zxy(z_deg_step4, 0, 0)
        R_step4_combined = R_step4_xy @ R_step4_yz
        add_measurements(
            R_step4_combined,
            ["+y", "-y", "+z", "-z"],
            f"step4_yz_x{tilt_x_deg_step4}_xy_z{z_deg_step4}",
        )


# ============================================================
# Samla data
# ============================================================
points = np.array(points)
dirs = np.array(dirs)
k_meas = np.linalg.norm(points, axis=1)
rot_ids = np.array(rot_ids)
colors = np.array(colors)


# ============================================================
# Tensor-fit: k(n) = nᵀ K n
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
expected_points = 6 + len(tilt_y_angles) * \
    (4 + 6 * len(z_angles)) + 4 + 4 * \
    len(z_angles_step3) + len(tilt_x_angles_step4) * \
    (4 + 4 * len(z_angles_step4))
print("Total points:", len(points), f"(expected {expected_points})")

rot_order = list(dict.fromkeys(rot_ids.tolist()))
for rot_label in rot_order:
    mask = rot_ids == rot_label
    unique_faces = sorted(set(face_ids[i] for i in np.where(mask)[0]))
    print(f"{rot_label}: {np.sum(mask)} pkt, faces={unique_faces}")

coef, *_ = np.linalg.lstsq(A, k_meas, rcond=None)

K_fit = np.array([
    [coef[0], coef[1], coef[2]],
    [coef[1], coef[3], coef[4]],
    [coef[2], coef[4], coef[5]],
])


# ============================================================
# Plot
# ============================================================
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

markers = ["o", "^", "s", "D", "P", "X"]

# Återanvänd överlappande ±y-värden i visualiseringen: plotta bara första unika.
plot_mask = np.ones(len(points), dtype=bool)
seen_y_keys = set()
for i, (p, face) in enumerate(zip(points, face_ids)):
    if face in ["+y", "-y"]:
        y_key = tuple(np.round(p, 8))
        if y_key in seen_y_keys:
            plot_mask[i] = False
        else:
            seen_y_keys.add(y_key)

points_plot = points[plot_mask]
rot_ids_plot = rot_ids[plot_mask]
face_ids_plot = np.array(face_ids)[plot_mask]
colors_plot = colors[plot_mask]

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
        points_plot[mask, 0],
        points_plot[mask, 1],
        points_plot[mask, 2],
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
for p, rot_label, face in zip(points_plot, rot_ids_plot, face_ids_plot):
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
