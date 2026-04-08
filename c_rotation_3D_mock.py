"""Continuum assumption test in 3D using a mock values for K.

1) Run mock K values for several model rotations.
2) Collect directional conductivity measurements.
3) Fit a symmetric 3D conductivity tensor K.
4) Evaluate fit quality (RMSE) and visualize measurement cloud + fitted ellipsoid.
"""

# Import important packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import time
import pyvista as pv

# File paths for saving results and images
ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv_files"
TENSOR_EXPORT_PATH = CSV_DIR / "tensor_sim_one.csv"
ROTATIONDATA_EXPORT_PATH = CSV_DIR / "rotationdata.csv"
IMAGES_DIR = ROOT / "images"

# Creating rotation matrixes for 3D rotations in z-x-y order
def rotation_matrix_zxy(z_deg, x_deg, y_deg):
    
    # converting degrees to radians for each rotation angle
    z_rad = np.radians(z_deg)
    x_rad = np.radians(x_deg)
    y_rad = np.radians(y_deg)

    cz, sz = np.cos(z_rad), np.sin(z_rad)
    cx, sx = np.cos(x_rad), np.sin(x_rad)
    cy, sy = np.cos(y_rad), np.sin(y_rad)

# Rotation matrixes for each axis (applied in z-x-y order)
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

# Function to compute directional conductivity k(n) = n^T K n for unit direction n
def directional_k(k_tensor, direction_vec):
    """Directional conductivity k(n) = n^T K n for unit direction n."""
    n = np.asarray(direction_vec, dtype=float)
    n = n / np.linalg.norm(n)
    return float(n.T @ k_tensor @ n)


# Rotation set in 3D: (z, x, y) in degrees
# 1) z from 0..75 at fixed x,y
# 2) then next x-step and sweep z again
# 3) then next y-step and repeat

angles_deg = [0, 15, 30, 45, 60, 75]    # rotation angles for each axis
rotations_3d = [
    (z_deg, x_deg, y_deg)
    for y_deg in angles_deg
    for x_deg in angles_deg
    for z_deg in angles_deg
]
print(
    f"Total rotations: {len(rotations_3d)} (expected {len(angles_deg) ** 3})")


# Results collector
k_results = {}

# Helper function to store results from one rotation and print summary
def add_rotation(rotation_deg, k_x_pos, k_x_neg, k_y_pos, k_y_neg, k_z_pos, k_z_neg):
    """Store 6 K values from one 3D rotation."""
    k_results[rotation_deg] = (
        k_x_pos, k_x_neg, k_y_pos, k_y_neg, k_z_pos, k_z_neg) # Store results for all 6 directiosn in each rotation 
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
def mock_dfn(rotation_deg):
    k_iso = 1.5
    return (k_iso, k_iso, k_iso, k_iso, k_iso, k_iso)


# Run mock DFN for all rotations and store results
for rot in rotations_3d:
    add_rotation(rot, *mock_dfn(rot))

assert set(k_results.keys()) == set(
    rotations_3d), "Missing one or more 3D rotations."


# Map results to global 3D vectors for plotting/fitting
points_x_pos = []
points_x_neg = []
points_y_pos = []
points_y_neg = []
points_z_pos = []
points_z_neg = []
measurement_points = []
measurement_directions = []
measurement_rotations = []
rotation_data_rows = []
dir_records = {"+x": [], "-x": [], "+y": [], "-y": [], "+z": [], "-z": []}

for rot in rotations_3d:
    k_x_pos, k_x_neg, k_y_pos, k_y_neg, k_z_pos, k_z_neg = k_results[rot]
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

    measurement_entries = [
        ("+x", k_x_pos * ex, k_x_pos),
        ("-x", k_x_neg * (-ex), k_x_neg),
        ("+y", k_y_pos * ey, k_y_pos),
        ("-y", k_y_neg * (-ey), k_y_neg),
        ("+z", k_z_pos * ez, k_z_pos),
        ("-z", k_z_neg * (-ez), k_z_neg),
    ]
    for direction_label, point, k_dir in measurement_entries:
        dir_records[direction_label].append((rot, k_dir, point))

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

direction_order = ["+x", "-x", "+y", "-y", "+z", "-z"]
for direction_label in direction_order:
    for rot, k_dir, point in dir_records[direction_label]:
        measurement_points.append(point)
        measurement_directions.append(direction_label)
        measurement_rotations.append(rot)
        rotation_data_rows.append(
            [
                len(rotation_data_rows) + 1,
                rot[0],
                rot[1],
                rot[2],
                direction_label,
                k_dir,
                point[0],
                point[1],
                point[2],
            ]
        )

measurement_points = np.array(measurement_points)
measurement_directions = np.array(measurement_directions)
measurement_rotations = np.array(measurement_rotations)

all_points = measurement_points


def _format_max_4_decimals(value):
    rounded = round(float(value), 4)
    text = f"{rounded:.4f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def show_rotation_box_interactive(rotations):
    """Show a transparent rotating box and let user jump to rotation ID."""
    if len(rotations) == 0:
        print("No rotations available, skipping interactive rotation box.")
        return

    box_half_size = 1.0
    base_vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    ) * box_half_size
    face_indices = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ]
    face_colors = [
        (0.1, 0.4, 1.0, 0.26),  # blue pair
        (0.1, 0.4, 1.0, 0.26),
        (0.1, 0.7, 0.2, 0.26),  # green pair
        (0.1, 0.7, 0.2, 0.26),
        (0.9, 0.2, 0.2, 0.26),  # red pair
        (0.9, 0.2, 0.2, 0.26),
    ]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Interactive Rotation Box (Origin at Center)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    axis_lim = box_half_size * 1.8
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    ax.set_box_aspect([1, 1, 1])

    # Keep origin visible in the middle of the box.
    ax.scatter([0.0], [0.0], [0.0], color="black", s=35)
    ax.text2D(0.02, 0.98, "Origin", transform=ax.transAxes, fontsize=10)

    box_collection = Poly3DCollection(
        [],
        facecolors=face_colors,
        edgecolors="navy",
        linewidths=1.0,
    )
    ax.add_collection3d(box_collection)

    status_text = ax.text2D(
        0.02,
        0.90,
        "",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
    )

    fig.subplots_adjust(bottom=0.16)
    box_ax = fig.add_axes([0.70, 0.05, 0.26, 0.05])
    rotation_id_box = TextBox(box_ax, "Rotation ID", initial="1")
    play_ax = fig.add_axes([0.53, 0.05, 0.14, 0.05])
    play_button = Button(play_ax, "Play")

    state = {
        "index": 0,
        "playing": False,
        "pause_until": 0.0,
    }

    def _update(frame_index):
        state["index"] = int(frame_index)
        z_deg, x_deg, y_deg = rotations[frame_index]
        r = rotation_matrix_zxy(z_deg, x_deg, y_deg)
        rotated_vertices = (r @ base_vertices.T).T
        rotated_faces = [[rotated_vertices[idx]
                          for idx in face] for face in face_indices]
        box_collection.set_verts(rotated_faces)
        status_text.set_text(
            f"Rotation {frame_index + 1}/{len(rotations)}\\n"
            f"z={z_deg} deg, x={x_deg} deg, y={y_deg} deg"
        )
        fig.canvas.draw_idle()

    def _submit_rotation_id(text):
        try:
            rotation_id = int(str(text).strip())
        except ValueError:
            rotation_id = 1
        rotation_id = max(1, min(len(rotations), rotation_id))
        if str(rotation_id) != str(text).strip():
            rotation_id_box.set_val(str(rotation_id))
        _update(rotation_id - 1)

    def _toggle_play(_event):
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")
        fig.canvas.draw_idle()

    def _tick():
        if not state["playing"]:
            return

        now = time.monotonic()
        if now < state["pause_until"]:
            return

        next_index = state["index"] + 1
        if next_index >= len(rotations):
            state["playing"] = False
            play_button.label.set_text("Play")
            fig.canvas.draw_idle()
            return

        _update(next_index)

        # Pause after each block of 36 rotations before continuing.
        if ((next_index + 1) % 36 == 0) and (next_index + 1 < len(rotations)):
            state["pause_until"] = now + 1.5

    rotation_id_box.on_submit(_submit_rotation_id)
    play_button.on_clicked(_toggle_play)
    _update(0)

    timer = fig.canvas.new_timer(interval=250)
    timer.add_callback(_tick)
    timer.start()

    plt.show()


CSV_DIR.mkdir(parents=True, exist_ok=True)
with open(ROTATIONDATA_EXPORT_PATH, "w", encoding="utf-8") as csv_file:
    csv_file.write(
        "point_id,z_deg,x_deg,y_deg,direction,k_dir,kx,ky,kz\n"
    )
    for row in rotation_data_rows:
        point_id = int(row[0])
        direction_label = str(row[4])
        numeric_values = [_format_max_4_decimals(value) for value in row[1:4]]
        numeric_values.extend(_format_max_4_decimals(value)
                              for value in row[5:9])
        csv_file.write(
            ",".join([str(point_id), *numeric_values[:3],
                     direction_label, *numeric_values[3:]])
            + "\n"
        )

print(f"Saved per-rotation measurement data to: {ROTATIONDATA_EXPORT_PATH}")


# Plot 1: raw directional measurements (3D)
fig1 = plt.figure(figsize=(8, 7))
ax1 = fig1.add_subplot(111, projection="3d")
measurement_order = np.arange(1, len(measurement_points) + 1)
measurement_scatter = ax1.scatter(
    measurement_points[:, 0],
    measurement_points[:, 1],
    measurement_points[:, 2],
    c=measurement_order,
    cmap="turbo",
    s=34,
    marker="o",
    alpha=0.92,
    edgecolors="none",
)
highlight_scatter = ax1.scatter(
    [measurement_points[0, 0]],
    [measurement_points[0, 1]],
    [measurement_points[0, 2]],
    s=120,
    marker="o",
    facecolors="none",
    edgecolors="black",
    linewidths=1.8,
)
ax1.set_xlabel("Kx")
ax1.set_ylabel("Ky")
ax1.set_zlabel("Kz")
ax1.set_title("Directional Hydraulic Conductivity")
ax1.text2D(
    0.02,
    0.98,
    "Color = Measurements 1-1296",
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
)

info_text = ax1.text2D(
    0.02,
    0.88,
    "",
    transform=ax1.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
)

fig1.subplots_adjust(bottom=0.16)
textbox_ax = fig1.add_axes([0.72, 0.07, 0.24, 0.045])
point_id_box = TextBox(
    ax=textbox_ax,
    label="Point ID",
    initial="1",
)


def set_measurement_focus(index):
    point = measurement_points[index]
    direction_label = measurement_directions[index]
    rotation = measurement_rotations[index]

    highlight_scatter._offsets3d = ([point[0]], [point[1]], [point[2]])
    info_text.set_text(
        f"Point {index + 1}/1296\n"
        f"Direction: {direction_label}\n"
        f"Rotation (z, x, y): {tuple(rotation)}\n"
        f"K = ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"
    )
    fig1.canvas.draw_idle()


def update_measurement_focus(order_number):
    index = int(order_number) - 1
    if index < 0 or index >= len(measurement_points):
        return
    set_measurement_focus(index)


def update_measurement_from_box(text):
    try:
        order_number = int(str(text).strip())
    except ValueError:
        point_id_box.set_val("")
        point_id_box.set_val("1")
        set_measurement_focus(0)
        return

    order_number = max(1, min(len(measurement_points), order_number))
    point_id_box.set_val(str(order_number))
    set_measurement_focus(order_number - 1)


point_id_box.on_submit(update_measurement_from_box)
update_measurement_focus(1)

lim = np.max(np.abs(all_points)) * 1.15
ax1.set_xlim(-lim, lim)
ax1.set_ylim(-lim, lim)
ax1.set_zlim(-lim, lim)
ax1.set_box_aspect([1, 1, 1])
fig1.colorbar(
    measurement_scatter,
    ax=ax1,
    pad=0.08,
    shrink=0.75,
    label="Measurements",
)


# Fit symmetric 3D conductivity tensor via least squares
# k_i = n_i^T K n_i
# K = [[Kxx, Kxy, Kxz],
#      [Kxy, Kyy, Kyz],
#      [Kxz, Kyz, Kzz]]
dirs = []
k_meas = []

for rot in rotations_3d:
    k_x_pos, k_x_neg, k_y_pos, k_y_neg, k_z_pos, k_z_neg = k_results[rot]
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


def build_continuous_shell_mesh(points_3d, local_fan_neighbors=6):
    """Build shell mesh using the same robust workflow as test_tva.

    Method: cloud -> delaunay_3d -> extract_surface -> triangulate.
    """
    if pv is None:
        return None, None, None

    points = np.asarray(points_3d, dtype=float)
    if len(points) < 4:
        return None, None, None

    cloud = pv.PolyData(points)

    try:
        vol = cloud.delaunay_3d()
        surface = vol.extract_surface().triangulate()
    except Exception:
        return None, None, None

    if surface is None or surface.n_cells == 0:
        return None, None, None

    point_ids = np.arange(1, len(points) + 1, dtype=int)
    mesh = surface
    return points, mesh, point_ids


def plot_pyvista_shell_with_point_ids(points, mesh, point_ids, output_path):
    """Plot all points, dense faces, and allow click-to-read point IDs."""
    if pv is None:
        print("PyVista not available: skipping nodes+faces surface plot.")
        return

    point_cloud = pv.PolyData(points)
    point_cloud["point_id"] = point_ids
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _pick_callback(picked_point):
        if picked_point is None:
            return
        picked_point = np.asarray(picked_point, dtype=float)
        nearest_index = int(
            np.argmin(np.linalg.norm(points - picked_point, axis=1)))
        print(
            f"Picked point_id={point_ids[nearest_index]} at "
            f"({points[nearest_index, 0]:.6f}, {points[nearest_index, 1]:.6f}, {points[nearest_index, 2]:.6f})"
        )

    def _add_scene(plotter):
        plotter.add_mesh(mesh, color="lightsteelblue",
                         opacity=0.30, show_edges=True)
        plotter.add_mesh(
            point_cloud,
            scalars="point_id",
            cmap="turbo",
            point_size=8,
            render_points_as_spheres=True,
            pickable=True,
        )
        plotter.add_scalar_bar(title="Point ID")
        plotter.add_axes()
        plotter.add_title(
            f"Mock DFN Dense Face Shell With Point IDs 1-{len(point_ids)}")
        plotter.add_text(
            "Högerklicka på en punkt för att skriva ut point_id",
            position="upper_left",
            font_size=10,
        )
        plotter.camera_position = "iso"
        try:
            plotter.enable_point_picking(
                callback=_pick_callback,
                show_message=True,
                color="yellow",
                font_size=10,
                use_mesh=False,
                show_point=True,
            )
        except Exception:
            print("Point picking is not available in this PyVista environment.")

    try:
        plotter = pv.Plotter(window_size=(1200, 800), off_screen=False)
        _add_scene(plotter)
        plotter.show(auto_close=False)
        plotter.screenshot(str(output_path))
        plotter.close()
    except Exception:
        # Fallback for headless environments.
        plotter = pv.Plotter(window_size=(1200, 800), off_screen=True)
        _add_scene(plotter)
        plotter.screenshot(str(output_path))
        plotter.close()

    print(f"Saved PyVista shell plot: {output_path}")


# Plot 3: PyVista continuous shell with all 1296 points retained
nodes, mesh, point_ids = build_continuous_shell_mesh(all_points)
if nodes is None:
    print("Could not build continuous shell mesh from point cloud; skipping Plot 3.")
else:
    print("\nPyVista continuous shell mesh summary:")
    print(f"  Total measurement points: {len(all_points)}")
    print(f"  Nodes used: {len(nodes)}")
    print(f"  Point IDs: {point_ids[0]}..{point_ids[-1]}")
    print(f"  Faces: {mesh.n_cells}")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_pyvista_shell_with_point_ids(
        nodes,
        mesh,
        point_ids,
        IMAGES_DIR / "mock_dfn_nodes_faces_surface.png",
    )


show_rotation_box_interactive(rotations_3d)
