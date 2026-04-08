"""Continuum assumption test in 3D using a mock DFN.

Concept mirrors the 2D workflow:
1) Run mock DFN for several model rotations.
2) Collect directional conductivity measurements.
3) Fit a symmetric 3D conductivity tensor K.
4) Evaluate fit quality (RMSE) and visualize measurement cloud + fitted ellipsoid.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from pathlib import Path

try:
    import pyvista as pv
except ImportError:  # pragma: no cover
    pv = None


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv_files"
TENSOR_EXPORT_PATH = CSV_DIR / "tensor_sim_one.csv"
ROTATIONDATA_EXPORT_PATH = CSV_DIR / "rotationdata.csv"
IMAGES_DIR = ROOT / "images"


def rotation_matrix_zxy(z_deg, x_deg, y_deg): 
    """Return local-to-global rotation for sequence z -> x -> y.

    Input tuple convention in this script is (z_deg, x_deg, y_deg).
    """
    z_rad = np.radians(z_deg) # Rotation around local z-axis (first)
    x_rad = np.radians(x_deg) # Rotation around local x-axis (second)
    y_rad = np.radians(y_deg) # Rotation around local y-axis (third)

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
    return float(n.T @ k_tensor @ n) # return scalar directional conductivity


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


# DFN results collector
dfn_results = {}

# Underlying "true" anisotropic tensor used by mock DFN
np.random.seed(42)
true_principal = np.array([6.0, 3.0, 1.0])
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
# def mock_dfn(rotation_deg):
#     k_iso = 1.5
#     return (k_iso, k_iso, k_iso, k_iso, k_iso, k_iso)


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
measurement_points = []
measurement_directions = []
measurement_rotations = []
rotation_data_rows = []

# This loop converts the raw scalar K measurements into 3D vectors in the global coordinate frame, ready for plotting and fitting
for rot in rotations_3d:
    k_x_pos, k_x_neg, k_y_pos, k_y_neg, k_z_pos, k_z_neg = dfn_results[rot] # 6 scalar K values for this rotation
    # Build the rotation matrix for this rotation and extract the global directions of local +/-x, +/-y, +/-z
    r = rotation_matrix_zxy(*rot)
    ex = r[:, 0]
    ey = r[:, 1]
    ez = r[:, 2]
    # Convert each scalar K measurement into a 3D point by multiplying the scalar K with the corresponding global direction vector
    # This gives us the actual directional conductivity vector in global coordinates for each measurement
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
ax1.set_title("Directional Hydraulic Conductivity (3D mock DFN)")
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


def _on_click(event):
    """Click on a point in the 3D scatter to select it."""
    if event.inaxes is not ax1 or event.button != 1:
        return
    # Project all 3D points to 2D screen coordinates
    from mpl_toolkits.mplot3d import proj3d
    proj = np.array([
        proj3d.proj_transform(p[0], p[1], p[2], ax1.get_proj())[:2]
        for p in measurement_points
    ])
    # Convert click to data coords in the projected space
    click = np.array([event.xdata, event.ydata])
    if click[0] is None:
        return
    dists = np.linalg.norm(proj - click, axis=1)
    nearest = int(np.argmin(dists))
    point_id_box.set_val(str(nearest + 1))
    set_measurement_focus(nearest)


fig1.canvas.mpl_connect("button_press_event", _on_click)

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
print(f"  Principal axes (columns):\n{eigvecs}")
print(f"  Angle between k1 axis and global x-axis: "
      f"{np.degrees(np.arccos(np.clip(abs(eigvecs[0, 0]), 0.0, 1.0))):.2f}°")
print(f"  Angle between k1 axis and global y-axis: "
      f"{np.degrees(np.arccos(np.clip(abs(eigvecs[1, 0]), 0.0, 1.0))):.2f}°")
print(f"  Angle between k1 axis and global z-axis: "
      f"{np.degrees(np.arccos(np.clip(abs(eigvecs[2, 0]), 0.0, 1.0))):.2f}°")
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
#Conductivity representation surface (or Lamé's stress ellipsoid analog for tensors)
# Sphere → multiply each point by K(n) → ellipsoid
u = np.linspace(0.0, 2.0 * np.pi, 64) # azimuthal angle for ellipsoid surface sampling (like longitude on a globe)
v = np.linspace(0.0, np.pi, 32) # polar angle for ellipsoid surface sampling (like latitude on a globe)
uu, vv = np.meshgrid(u, v) # create a grid of angles for sampling the ellipsoid surface

# Unit sphere parametric points 
sx = np.cos(uu) * np.sin(vv)
sy = np.sin(uu) * np.sin(vv)
sz = np.cos(vv)

# Build conductivity representation surface: K(n)*n in global frame.
# K(n) = n^T K_fit n; plotted as K(n)*n -- this is the same surface the measurement points trace, so the fitted surface will pass through them.
k_surf = (
    sx**2 * kxx + sy**2 * kyy + sz**2 * kzz
    + 2.0 * sx * sy * kxy
    + 2.0 * sx * sz * kxz
    + 2.0 * sy * sz * kyz
)
# Scale the unit sphere by the directional conductivity to get the actual 3D coordinates of the fitted ellipsoid surface in global frame
xe = k_surf * sx # actual 3D coordinates of the fitted ellipsoid surface in global frame
ye = k_surf * sy
ze = k_surf * sz

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
    """Build shell mesh so that ALL 1296 points appear in a face.

    Strategy: normalise every point to the unit sphere, then compute the
    convex hull of the unit directions.  Because all unit vectors lie on
    the sphere surface they are all hull vertices, so every original point
    is guaranteed to appear in at least one triangle.  The face indices are
    then applied back to the original (non-normalised) 3-D coordinates.
    """
    from scipy.spatial import ConvexHull

    if pv is None:
        return None, None, None

    points = np.asarray(points_3d, dtype=float)
    if len(points) < 4:
        return None, None, None

    # Normalise to unit sphere so every point is on the convex boundary
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    unit = points / np.where(norms > 0, norms, 1.0)

    try:
        hull = ConvexHull(unit)
    except Exception:
        return None, None, None

    faces = hull.simplices          # triangle indices into `unit` == into `points`

    # Build PyVista face array: [3, i0, i1, i2,  3, i0, i1, i2, ...]
    pv_faces = np.hstack([
        np.full((len(faces), 1), 3, dtype=np.intp),
        faces,
    ]).ravel()

    # Use original (non-normalised) points for the mesh geometry
    mesh = pv.PolyData(points, pv_faces)
    point_ids = np.arange(1, len(points) + 1, dtype=int)
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


def plot_pyvista_fitted_surface(k_fit, eigvals, eigvecs, all_points, output_path):
    """Plot 4: fitted conductivity surface K(n)*n as a PyVista mesh.

    The surface is built from a dense uniform sphere sampling so the mesh
    is smooth and independent of the measurement grid.  The raw measurement
    points are overlaid for comparison.
    """
    if pv is None:
        print("PyVista not available: skipping fitted surface plot.")
        return

    # Dense unit-sphere grid (Fibonacci lattice -> uniform coverage)
    n_pts = 4000
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    ii = np.arange(n_pts)
    theta_fib = np.arccos(1.0 - 2.0 * (ii + 0.5) / n_pts)
    phi_fib   = 2.0 * np.pi * ii / golden
    sx = np.sin(theta_fib) * np.cos(phi_fib)
    sy = np.sin(theta_fib) * np.sin(phi_fib)
    sz = np.cos(theta_fib)
    unit_dirs = np.column_stack([sx, sy, sz])   # (n_pts, 3)

    # Evaluate K(n) = n^T K_fit n for every direction
    kxx, kxy, kxz = k_fit[0, 0], k_fit[0, 1], k_fit[0, 2]
    kyy, kyz, kzz = k_fit[1, 1], k_fit[1, 2], k_fit[2, 2]
    k_vals = (
        unit_dirs[:, 0]**2 * kxx
        + unit_dirs[:, 1]**2 * kyy
        + unit_dirs[:, 2]**2 * kzz
        + 2.0 * unit_dirs[:, 0] * unit_dirs[:, 1] * kxy
        + 2.0 * unit_dirs[:, 0] * unit_dirs[:, 2] * kxz
        + 2.0 * unit_dirs[:, 1] * unit_dirs[:, 2] * kyz
    )
    surf_pts = unit_dirs * k_vals[:, np.newaxis]   # K(n)*n

    # Convex hull on unit directions -> triangulation covering all pts
    from scipy.spatial import ConvexHull
    hull = ConvexHull(unit_dirs)
    faces = hull.simplices
    pv_faces = np.hstack([
        np.full((len(faces), 1), 3, dtype=np.intp),
        faces,
    ]).ravel()

    surf_mesh = pv.PolyData(surf_pts, pv_faces)
    surf_mesh["K"] = k_vals

    meas_cloud = pv.PolyData(np.asarray(all_points, dtype=float))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _add_scene(plotter):
        plotter.add_mesh(
            surf_mesh,
            scalars="K",
            cmap="coolwarm",
            opacity=0.55,
            show_edges=False,
            label="Fitted K(n) surface",
        )
        plotter.add_mesh(
            meas_cloud,
            color="white",
            point_size=4,
            render_points_as_spheres=True,
            opacity=0.6,
            label="Measurements",
        )
        # Principal axes as arrows
        colors = ["red", "green", "blue"]
        labels = [f"k1={eigvals[0]:.2f}", f"k2={eigvals[1]:.2f}", f"k3={eigvals[2]:.2f}"]
        for i, (col, lbl) in enumerate(zip(colors, labels)):
            tip = eigvals[i] * eigvecs[:, i]
            arrow = pv.Arrow(start=(0, 0, 0), direction=tip, scale=np.linalg.norm(tip))
            plotter.add_mesh(arrow, color=col, label=lbl)
        plotter.add_scalar_bar(title="K(n)  [m/d]")
        plotter.add_axes()
        plotter.add_title("Fitted Conductivity Surface K(n)·n  (PyVista)")
        plotter.camera_position = "iso"

    try:
        plotter = pv.Plotter(window_size=(1200, 800), off_screen=False)
        _add_scene(plotter)
        plotter.show(auto_close=False)
        plotter.screenshot(str(output_path))
        plotter.close()
    except Exception:
        plotter = pv.Plotter(window_size=(1200, 800), off_screen=True)
        _add_scene(plotter)
        plotter.screenshot(str(output_path))
        plotter.close()

    print(f"Saved fitted surface plot: {output_path}")


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

# Plot 4: PyVista fitted conductivity surface K(n)*n
if pv is not None:
    plot_pyvista_fitted_surface(
        k_fit,
        eigvals,
        eigvecs,
        all_points,
        IMAGES_DIR / "mock_dfn_fitted_surface.png",
    )
