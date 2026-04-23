"""
Notes
-----
This is an example of a model.
"""

import datetime
import os
import sys
from pathlib import Path
import logging
from unittest import result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import andfn
from andfn import ConstantHeadLine

logging.basicConfig(level=logging.INFO)


# ======================================================
# Logger
# ======================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        self.newline = True

    def write(self, message):
        if not message:
            return

        out = ""
        for char in message:
            if self.newline and char != '\n':
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                out += f"{now} INFO "
                self.newline = False
            out += char
            if char == '\n':
                self.newline = True

        self.terminal.write(out)
        self.log.write(out)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ======================================================
# Rotation helpers
# ======================================================
def apply_rotations(regbox, rotations):
    for angle, axis in rotations:
        regbox.rotate(angle=angle, axis=axis)


def rotate_vector(v, rotations):
    v = np.asarray(v, float)
    for angle, axis in rotations:
        axis = np.asarray(axis, float)
        axis /= np.linalg.norm(axis)
        theta = np.deg2rad(angle)
        v = (
            v * np.cos(theta)
            + np.cross(axis, v) * np.sin(theta)
            + axis * np.dot(axis, v) * (1.0 - np.cos(theta))
        )
    return v


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    # save figure
    save = True
    scale = 1
    tracking = True
    animate = False
    block_on_final_k_plot = True

    ncoef = 10 * 0 + 10
    nint = ncoef * 2

    start0 = datetime.datetime.now()

    sim_dir = Path(
        r"C:\Users\SEMB94861\Flopy\flopythesis\simulations"
    ) / start0.strftime("%Y%m%d_%H%M%S")
    sim_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = Logger(sim_dir / "terminal_output.txt")

    print(f"\n---- IMPORT DFN ----")
    print(f"Program started at {start0}")
    print(f"Simulation output directory: {sim_dir}")

    # --------------------------------------------------
    # Load DFN
    # --------------------------------------------------
    t_load_start = datetime.datetime.now()
    dfn_org = andfn.DFN("DFN test FracMan", discharge_int=50)

    path = os.path.join(
        r"C:\Users\SEMB94861\Flopy\flopythesis\csv_files",
        "40000.csv",
    )

    fracture_import_kwargs = dict(
        radius_str="EquivRadius[m]",
        x_str="FractureX[m]",
        y_str="FractureY[m]",
        z_str="FractureZ[m]",
        t_str="Transmissivity[m2/s]",
        trend_str="Trend[deg]",
        plunge_str="Plunge[deg]",
        e_str="Aperture[m]",
        remove_tolerance=1e-3,
        remove_isolated=False,
    )

    dfn_org.import_fractures_from_file(path, **fracture_import_kwargs)
    print(f"Time to load DFN: {datetime.datetime.now() - t_load_start}")

    # --------------------------------------------------
    # Rotation configurations
    # --------------------------------------------------
    rotation_configs = []

    rotation_configs.append(([], ["x", "y", "z"], "start"))

    for z in range(15, 90, 15):
        rotation_configs.append(
            ([(z, (0, 0, 1))], ["x", "y"], f"base_z{z}")
        )

    for ty in range(15, 90, 15):
        rotation_configs.append(
            ([(ty, (0, 1, 0))], ["x", "z"], f"tilt_y{ty}")
        )
        for z in range(15, 90, 15):
            rotation_configs.append(
                (
                    [(ty, (0, 1, 0)), (z, (0, 0, 1))],
                    ["x", "z"],
                    f"tilt_y{ty}_z{z}",
                )
            )

    for tx in range(15, 90, 15):
        rotation_configs.append(
            ([(tx, (1, 0, 0))], ["y", "z"], f"step3_yz_x{tx}")
        )
        for z in range(15, 90, 15):
            rotation_configs.append(
                (
                    [(tx, (1, 0, 0)), (z, (0, 0, 1))],
                    ["y", "z"],
                    f"step3_yz_x{tx}_xy_z{z}",
                )
            )

    # --------------------------------------------------
    # Plot & bookkeeping
    # --------------------------------------------------
    plot_points = []
    plot_dirs = []
    plot_colors = []
    plot_labels = []
    plot_face_ids = []

    FACE_COLORS = {
        "+x": "#eb0a1c", "-x": "#d00000",
        "+y": "#457b9d", "-y": "#1d3557",
        "+z": "#2a9d8f", "-z": "#1b7f6b",
    }

    axis_faces = {
        "x": ("left", "right"),
        "y": ("front", "back"),
        "z": ("top", "bottom"),
    }

    head0, head1 = 200, 100

    dfn_plotters = {}

    # --------------------------------------------------
    # Main simulation loop
    # --------------------------------------------------
    for idx, (rotations, active_axes, rot_label) in enumerate(rotation_configs):
        t_rot_start = datetime.datetime.now()
        print(f"\n[{idx+1}/{len(rotation_configs)}] === Rotation {rot_label} ===")

        for axis_name in active_axes:
            t_axis_start = datetime.datetime.now()
            face_low, face_high = axis_faces[axis_name]

            dfn = andfn.DFN("Copy", discharge_int=50)
            #dfn.import_fractures_from_file(path, **fracture_import_kwargs)
            dfn.add_fracture(list(dfn_org.fractures))

            regbox = andfn.RectangularRegion(
                label="box",
                center=[0, 0, 0],
                x_vec=[1, 0, 0],
                y_vec=[0, 1, 0],
                z_vec=[0, 0, 1],
                xl=500,
                yl=500,
                zl=500,
            )

            apply_rotations(regbox, rotations)

            if axis_name == "x":
                L = regbox.xl
                A = regbox.yl * regbox.zl
                v0 = np.array([1, 0, 0])
                faces = ("+x", "-x")
            elif axis_name == "y":
                L = regbox.yl
                A = regbox.xl * regbox.zl
                v0 = np.array([0, 1, 0])
                faces = ("+y", "-y")
            else:
                L = regbox.zl
                A = regbox.xl * regbox.yl
                v0 = np.array([0, 0, 1])
                faces = ("+z", "-z")

            reg_fracs_in, reg_fracs_out = regbox.check_fractures(
                dfn.fractures, tree=dfn.tree
            )
            dfn.delete_fracture(reg_fracs_out)

            regbox.frac_intersections(dfn.fractures, face=face_low, head=head0)
            regbox.frac_intersections(
                dfn.fractures, face=face_high, head=head1)

            dfn.check_connectivity()
            # dfn.reset()
            dfn.set_kwargs(
                COEF_RATIO=0.001,
                MAX_ITERATIONS=30,
                MAX_NCOEF=200,
                MAX_ERROR=5e-4,
            )

            solve_failed = False
            try:
                t_solve_start = datetime.datetime.now()
                dfn.solve(unconsolidate=True)
                t_solve_end = datetime.datetime.now()
                print(
                    f"Time to solve axis {axis_name}: {t_solve_end - t_solve_start}")
            except RuntimeError as exc:
                solve_failed = True
                print(
                    f"Solver failed for rot={rot_label}, axis={axis_name}: {exc}")

            start2 = datetime.datetime.now()

            if solve_failed:
                sum_flows = np.nan
                k_axis = np.nan
            else:
                sum_flows = regbox.get_total_flow() / 2.0
                k_axis = sum_flows * L / (A * (head1 - head0))

            if np.isnan(k_axis):
                continue

            n_pos = rotate_vector(v0, rotations)
            n_neg = -n_pos

            plot_points.extend([k_axis * n_pos, k_axis * n_neg])
            plot_dirs.extend([n_pos, n_neg])
            plot_colors.extend([FACE_COLORS[faces[0]], FACE_COLORS[faces[1]]])
            plot_labels.extend([rot_label, rot_label])
            plot_face_ids.extend([faces[0], faces[1]])

            if not solve_failed:
                if save:
                    print(
                        f"\n---- SAVING DFN PLOT FOR ROTATION {idx+1} ({axis_name}-direction solve) ----")
                    p1 = dfn.initiate_plotter(
                        title=True, off_screen=True, scale=1, axis=True)

                    dfn.plot_fractures_head(
                        p1, 40, 10, opacity=1, contour=True
                    )

                    import pyvista as pv
                    box_mesh = pv.PolyData(regbox.vertices, regbox.faces)

                    # Colors mapping to actual PyVista output
                    face_color_list = [
                        FACE_COLORS["+z"],  # Top
                        FACE_COLORS["-z"],  # Bottom
                        FACE_COLORS["+y"],  # Front
                        FACE_COLORS["-y"],  # Back
                        FACE_COLORS["-x"],  # Left
                        FACE_COLORS["+x"]  # Right
                    ]

                    for i, face_indices in enumerate(regbox.faces.reshape(-1, 5)):
                        # Single face representation: [4, pt1, pt2, pt3, pt4]
                        single_face = np.hstack([[4], face_indices[1:]])
                        face_mesh = pv.PolyData(regbox.vertices, single_face)
                        p1.add_mesh(
                            face_mesh, color=face_color_list[i], opacity=0.3, show_edges=True)

                    rotations_dir = sim_dir / "rotations"
                    rotations_dir.mkdir(parents=True, exist_ok=True)
                    img_path = rotations_dir / \
                        f"dfn_plot_rot{idx+1}_{axis_name}.html"
                    # Save an interactive html of the PyVista plot
                    p1.export_html(img_path)

                    if str(idx+1) not in dfn_plotters:
                        dfn_plotters[str(idx+1)] = {}
                    # Keep path for potential later use
                    dfn_plotters[str(idx+1)][axis_name] = str(img_path)
                    p1.close()

            print(
                f"Total time for axis {axis_name}: {datetime.datetime.now() - t_axis_start}")
        print(
            f"Total time for rotation {rot_label}: {datetime.datetime.now() - t_rot_start}")

    # --------------------------------------------------
    # ===== PLOTTING & EXPORT =====
    # --------------------------------------------------
    t_plot_start = datetime.datetime.now()
    if plot_points:
        points = np.array(plot_points)
        dirs = np.array(plot_dirs)
        k_meas = np.linalg.norm(points, axis=1)
        rot_ids = np.array(plot_labels)
        colors = np.array(plot_colors)
        face_ids = np.array(plot_face_ids)

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

        K_fit, A, rmse_val = fit_tensor_least_squares(dirs, k_meas)

        print("Rank(A):", np.linalg.matrix_rank(A))

        expected_points = 6 + 4 * 4 + 5 * \
            (4 + 6 * 5) + 5 * (4 + 4 * 5)  # Est. from original logic
        print("Total points:", len(points), f"(expected ~{expected_points})")

        rot_order = list(dict.fromkeys(rot_ids.tolist()))
        for rot_label in rot_order:
            mask = rot_ids == rot_label
            unique_faces = sorted(set(face_ids[i] for i in np.where(mask)[0]))
            print(f"{rot_label}: {np.sum(mask)} pkt, faces={unique_faces}")

        print(
            f"RMSE residual (least squares): {rmse_val:.6e} (0 = perfect ellipsoid)")

        Kxx, Kxy, Kxz = K_fit[0, 0], K_fit[0, 1], K_fit[0, 2]
        Kyy, Kyz = K_fit[1, 1], K_fit[1, 2]
        Kzz = K_fit[2, 2]

        evals_print, evecs_print = np.linalg.eigh(K_fit)
        order_idx = np.argsort(evals_print)[::-1]
        eigvals = evals_print[order_idx]
        eigvecs = evecs_print[:, order_idx]

        print("\nFitted 3D conductivity tensor")
        print(f"  K = [[{Kxx:.4e}, {Kxy:.4e}, {Kxz:.4e}],")
        print(f"       [{Kxy:.4e}, {Kyy:.4e}, {Kyz:.4e}],")
        print(f"       [{Kxz:.4e}, {Kyz:.4e}, {Kzz:.4e}]]")
        print(
            f"  Principal values: k1 = {eigvals[0]:.4e}, k2 = {eigvals[1]:.4e}, k3 = {eigvals[2]:.4e}")
        print("  Principal axes:")
        print(
            f"    v1 = [{eigvecs[0, 0]:.4f}, {eigvecs[1, 0]:.4f}, {eigvecs[2, 0]:.4f}]")
        print(
            f"    v2 = [{eigvecs[0, 1]:.4f}, {eigvecs[1, 1]:.4f}, {eigvecs[2, 1]:.4f}]")
        print(
            f"    v3 = [{eigvecs[0, 2]:.4f}, {eigvecs[1, 2]:.4f}, {eigvecs[2, 2]:.4f}]\n")

        # ====== Export Rotations CSVs ======
        # Save only in simulations folder
        sim_dir.mkdir(parents=True, exist_ok=True)
        POINTS_CSV_PATH = sim_dir / "ny_rotation_points.csv"
        TENSOR_CSV_PATH = sim_dir / "ny_rotation_tensor_for_continuum.csv"

        header_points = "point_id;rotation_label;face;kx;ky;kz;k_value;dir_x;dir_y;dir_z\n"
        with open(POINTS_CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
            f.write(header_points)
            for i, (p, d, r_label, face) in enumerate(zip(points, dirs, rot_ids, face_ids), start=1):
                k_val = float(np.linalg.norm(p))
                f.write(
                    f"{i};{r_label};{face};"
                    f"{p[0]:.10g};{p[1]:.10g};{p[2]:.10g};"
                    f"{k_val:.10g};{d[0]:.10g};{d[1]:.10g};{d[2]:.10g}\n"
                )

        header_tensor = "angle_deg;k_xx;k_xy;k_xz;k_yx;k_yy;k_yz;k_zx;k_zy;k_zz\n"
        with open(TENSOR_CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
            f.write(header_tensor)
            f.write(
                "0.0;"
                f"{K_fit[0, 0]:.10g};{K_fit[0, 1]:.10g};{K_fit[0, 2]:.10g};"
                f"{K_fit[1, 0]:.10g};{K_fit[1, 1]:.10g};{K_fit[1, 2]:.10g};"
                f"{K_fit[2, 0]:.10g};{K_fit[2, 1]:.10g};{K_fit[2, 2]:.10g}\n"
            )
        print(f"Saved point CSV to: {POINTS_CSV_PATH}")
        print(f"Saved tensor CSV to: {TENSOR_CSV_PATH}")

        # ====== Plot 1: Ellipsoid Fit ======
        def build_ellipsoid_mesh_from_tensor(k_tensor, n_u=60, n_v=30):
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
            return ellipsoid[:, 0].reshape(x.shape), ellipsoid[:, 1].reshape(y.shape), ellipsoid[:, 2].reshape(z.shape), evals, evecs

        fig_fit = plt.figure(figsize=(8, 7))
        ax_fit = fig_fit.add_subplot(111, projection="3d")

        ex, ey, ez, evals_fit, evecs_fit = build_ellipsoid_mesh_from_tensor(
            K_fit)

        ax_fit.plot_surface(ex, ey, ez, color="#8ecae6",
                            alpha=0.28, linewidth=0)

        ax_fit.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors, s=40, alpha=0.8, edgecolors="black", linewidths=0.35,
            label="K magnitude"
        )

        axis_colors = ["#d62828", "#2a9d8f", "#1d3557"]
        for i in range(3):
            vec = evecs_fit[:, i] * evals_fit[i]
            ax_fit.quiver(0, 0, 0, vec[0], vec[1],
                          vec[2], color=axis_colors[i], linewidth=2)

        lim_fit = max(np.max(np.abs(points)) * 1.2,
                      np.max(np.abs(np.array([ex, ey, ez]))))
        ax_fit.set_xlim(-lim_fit, lim_fit)
        ax_fit.set_ylim(-lim_fit, lim_fit)
        ax_fit.set_zlim(-lim_fit, lim_fit)
        ax_fit.set_box_aspect([1, 1, 1])
        ax_fit.set_xlabel("K_x")
        ax_fit.set_ylabel("K_y")
        ax_fit.set_zlabel("K_z")
        ax_fit.set_title("Least-squares fitted conductivity ellipsoid")

        # Add tensor text box to plot
        tensor_text = (
            "Fitted Tensor [m/s]:\n"
            f"{K_fit[0, 0]:.2e}  {K_fit[0, 1]:.2e}  {K_fit[0, 2]:.2e}\n"
            f"{K_fit[1, 0]:.2e}  {K_fit[1, 1]:.2e}  {K_fit[1, 2]:.2e}\n"
            f"{K_fit[2, 0]:.2e}  {K_fit[2, 1]:.2e}  {K_fit[2, 2]:.2e}\n\n"
            f"RMSE: {rmse_val:.2e} m/s"
        )

        # Place text box in top right corner
        ax_fit.text2D(0.95, 0.95, tensor_text, transform=ax_fit.transAxes,
                      fontsize=10, verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig_fit.tight_layout()
        import pyvista as pv
        if save:
            fig_fit.savefig(sim_dir / "fitted_ellipsoid.png", dpi=300)

            p_ellipsoid = pv.Plotter(off_screen=True)
            grid = pv.StructuredGrid(ex, ey, ez)
            pts = pv.PolyData(points)
            p_ellipsoid.add_mesh(grid, color="lightblue", opacity=0.3)
            p_ellipsoid.add_points(pts, color="blue", point_size=10.0)
            p_ellipsoid.export_html(sim_dir / "fitted_ellipsoid.html")
            p_ellipsoid.close()

            # Save VTK formats
            grid.save(sim_dir / "fitted_ellipsoid_mesh.vtk")
            pts.save(sim_dir / "fitted_ellipsoid_points.vtk")

        # ====== Plot 2: K-values only ======
        if save:
            p_points_only = pv.Plotter(off_screen=True)
            pts = pv.PolyData(points)
            p_points_only.add_points(pts, color="blue", point_size=10.0)
            p_points_only.export_html(sim_dir / "k_values_points.html")
            p_points_only.close()

        # ====== Plot 3: Connected Mesh of K-values ======
        if save:
            try:
                p_mesh = pv.Plotter(off_screen=True)
                pts = pv.PolyData(points)
                # Delaunay 3D skapar ett volume mesh som vi extraherar surface från för att få ett "hölje"
                mesh = pts.delaunay_3d().extract_surface()
                p_mesh.add_mesh(mesh, color="lightblue",
                                opacity=0.5, show_edges=True)
                p_mesh.add_points(pts, color="blue", point_size=10.0)
                p_mesh.export_html(sim_dir / "k_values_mesh.html")
                p_mesh.close()

                # Spara ned nätet för k-värdena
                mesh.save(sim_dir / "k_values_connected_mesh.vtk")
            except Exception as e:
                print(
                    f"Kunde inte skapa ett sammanhängande nät av punkterna: {e}")

        print(
            f"Time to generate plots and CSVs: {datetime.datetime.now() - t_plot_start}")

        if block_on_final_k_plot:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(0.1)

    end = datetime.datetime.now()
    print(f"\nProgram ended at {end}")
    print(f"Time elapsed: {end - start0}")
