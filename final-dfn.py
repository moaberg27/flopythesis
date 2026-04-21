"""
Notes
-----
This is an example of a model.
"""

import datetime
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import andfn

# Configure logging
import logging

from andfn import ConstantHeadLine

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # save figure
    save = False
    scale = 1
    tracking = True
    animate = False
    block_on_dfn_plots = False
    block_on_final_k_plot = True

    ncoef = 10 * 0 + 10
    nint = ncoef * 2

    start0 = datetime.datetime.now()
    print("\n---- IMPORT DFN ----")
    print(f"Program started at {start0}")

    # load the geometry
    dfn_org = andfn.DFN("DFN test FracMan", discharge_int=50)

    # name ="p32_case11"
    path = os.path.join(
    r"C:\Users\SEVB94859\Flopy\flopythesis\fracs\\", "fracs_connected_properties.csv")
    reload = False
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

    # Check if it exist saved
    print("DFN  importing from file")
    dfn_org.import_fractures_from_file(path, **fracture_import_kwargs)

    # Rotation sequence from c_rotation setup
    tilt_y_angles = list(range(15, 90, 15))  # 15, 30, ..., 75
    z_angles = list(range(15, 90, 15))  # 15, 30, ..., 75
    z_angles_no_tilt = list(range(15, 90, 15))  # 15, 30, ..., 75
    tilt_x_angles_step3 = list(range(15, 90, 15))  # 15, 30, ..., 75
    z_angles_step3 = list(range(15, 90, 15))  # 15, 30, ..., 75

    def rotation_matrix_zxy(z_deg, x_deg, y_deg):
        z, x, y = np.radians([z_deg, x_deg, y_deg])
        cz, sz = np.cos(z), np.sin(z)
        cx, sx = np.cos(x), np.sin(x)
        cy, sy = np.cos(y), np.sin(y)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz,  cz, 0], [0,   0,  1]])
        return rx @ ry @ rz

    # Collect configurations to run
    rotation_configs = []

    # 1) Startläge: spara ±x, ±y, ±z
    rotation_configs.append(
        ({"z_deg": 0, "x_deg": 0, "y_deg": 0}, ["x", "y", "z"], "start"))

    # 2) Rotation utan tilt: rotera bara i xy-planet och spara ±x, ±y.
    for z_deg in z_angles_no_tilt:
        rotation_configs.append(
            ({"z_deg": z_deg, "x_deg": 0, "y_deg": 0}, ["x", "y"], f"base_z{z_deg}"))

    # 3) För varje tilt i xz-planet (rotation runt y): spara ±x, ±z
    for tilt_y_deg in tilt_y_angles:
        rotation_configs.append(({"z_deg": 0, "x_deg": 0, "y_deg": tilt_y_deg}, [
                                "x", "z"], f"tilt_y{tilt_y_deg}"))
        for z_deg in z_angles:
            rotation_configs.append(({"z_deg": z_deg, "x_deg": 0, "y_deg": tilt_y_deg}, [
                                    "x", "y", "z"], f"tilt_y{tilt_y_deg}_z{z_deg}"))

    # 4) Tredje steg: ny sekvens som startar om från normalläge och kör yz = 15, 30, 45, 60, 75 (rotation runt x).
    for tilt_x_deg in tilt_x_angles_step3:
        rotation_configs.append(({"z_deg": 0, "x_deg": tilt_x_deg, "y_deg": 0}, [
                                "y", "z"], f"step3_yz_x{tilt_x_deg}"))
        for z_deg in z_angles_step3:
            rotation_configs.append(({"z_deg": z_deg, "x_deg": tilt_x_deg, "y_deg": 0}, [
                                    "y", "z"], f"step3_yz_x{tilt_x_deg}_xy_z{z_deg}"))

    # Store points for 3D plotting
    dfn_plotters = {}  # Save all PyVista plots here to view later
    plot_points = []
    plot_dirs = []
    plot_colors = []
    plot_labels = []
    plot_face_ids = []

    head0 = 100
    head1 = 200
    axis_faces = {
        "x": ("left", "right"),
        "y": ("front", "back"),
        "z": ("top", "bottom"),
    }

    # Face colors like in c_rotation
    FACE_COLORS = {
        "+x": "#eb0a1c", "-x": "#d00000",
        "+y": "#457b9d", "-y": "#1d3557",
        "+z": "#2a9d8f", "-z": "#1b7f6b"
    }

    for idx, (angles, active_axes, rot_label) in enumerate(rotation_configs):
        z_deg = angles["z_deg"]
        x_deg = angles["x_deg"]
        y_deg = angles["y_deg"]
        print(
            f"\n[{idx+1}/{len(rotation_configs)}] === Rotation {rot_label}: z={z_deg}, x={x_deg}, y={y_deg} ===")

        # Calculate rotation matrix for global projection
        r_matrix = rotation_matrix_zxy(z_deg, x_deg, y_deg)

        for axis_name in active_axes:
            face_low, face_high = axis_faces[axis_name]

            dfn = andfn.DFN("Copy", discharge_int=50)
            dfn.import_fractures_from_file(path, **fracture_import_kwargs)

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

            L = regbox.xl if axis_name == "x" else (
                regbox.yl if axis_name == "y" else regbox.zl)
            if axis_name == "x":
                A = regbox.yl * regbox.zl
            elif axis_name == "y":
                A = regbox.xl * regbox.zl
            else:
                A = regbox.xl * regbox.yl

            # Rotate regbox using extrinsic (lab-frame) rotations
            # Apply rotations in reverse order of the matrix multiplication to get the same result
            if y_deg:
                regbox.rotate(angle=y_deg, axis=[0, 1, 0])
            if x_deg:
                regbox.rotate(angle=x_deg, axis=[1, 0, 0])
            if z_deg:
                regbox.rotate(angle=z_deg, axis=[0, 0, 1])
            
            # Standard face names for the rotated local axes
            face_low, face_high = axis_faces[axis_name]

            reg_fracs_in, reg_fracs_out = regbox.check_fractures(
                dfn.fractures, tree=dfn.tree)

            print(
                f"[{axis_name}] Fractures before clip: {len(dfn.fractures)}")
            dfn.delete_fracture(reg_fracs_out)
            print(
                f"[{axis_name}] Fractures after clip: {len(dfn.fractures)}")

            regbox.frac_intersections(
                dfn.fractures, face=face_low, head=head0)
            regbox.frac_intersections(
                dfn.fractures, face=face_high, head=head1)

            dfn.check_connectivity()
            dfn.check_connectivity()

            dfn.set_kwargs(COEF_RATIO=0.001, MAX_ITERATIONS=30,
                           MAX_NCOEF=200, MAX_ERROR=5e-4)

            start1 = datetime.datetime.now()
            print(f"\n---- SOLVE DFN ({axis_name}-direction) ----")
            solve_failed = False
            try:
                dfn.solve(unconsolidate=True)
            except RuntimeError as exc:
                solve_failed = True
                print(
                    f"Solver failed for rot={rot_label}, axis={axis_name}: {exc}")

            start2 = datetime.datetime.now()

            if solve_failed:
                sum_flows = np.nan
                k_axis = np.nan
            else:
                sum_flows = regbox.get_total_flow()/2
                k_axis = sum_flows * L / (A * (head1 - head0))

            print(f"[{axis_name}] Total flow: {sum_flows:.2e} m^3/s")
            print(f"[{axis_name}] k: {k_axis:.2e} m/s")

            # Skip if DFN is not connected (continue with next axis/rotation instead of exiting)
            if np.isnan(k_axis):
                print(f"WARNING: DFN not connected for rot={rot_label}, axis={axis_name}. Skipping this combination.")
                continue

            # Add to plotting structures like in c_rotation
            # We assume flow enters along the local axis positive and negative directions
            if axis_name == "x":
                n_pos = r_matrix @ np.array([1, 0, 0])
                n_neg = r_matrix @ np.array([-1, 0, 0])
                plot_points.append(k_axis * n_pos)
                plot_points.append(k_axis * n_neg)
                plot_dirs.extend([n_pos, n_neg])
                plot_colors.extend([FACE_COLORS["+x"], FACE_COLORS["-x"]])
                plot_labels.extend([rot_label, rot_label])
                plot_face_ids.extend(["+x", "-x"])
            elif axis_name == "y":
                n_pos = r_matrix @ np.array([0, 1, 0])
                n_neg = r_matrix @ np.array([0, -1, 0])
                plot_points.append(k_axis * n_pos)
                plot_points.append(k_axis * n_neg)
                plot_dirs.extend([n_pos, n_neg])
                plot_colors.extend([FACE_COLORS["+y"], FACE_COLORS["-y"]])
                plot_labels.extend([rot_label, rot_label])
                plot_face_ids.extend(["+y", "-y"])
            elif axis_name == "z":
                n_pos = r_matrix @ np.array([0, 0, 1])
                n_neg = r_matrix @ np.array([0, 0, -1])
                plot_points.append(k_axis * n_pos)
                plot_points.append(k_axis * n_neg)
                plot_dirs.extend([n_pos, n_neg])
                plot_colors.extend([FACE_COLORS["+z"], FACE_COLORS["-z"]])
                plot_labels.extend([rot_label, rot_label])
                plot_face_ids.extend(["+z", "-z"])

            if str(idx+1) not in dfn_plotters and not solve_failed:
                print(
                    f"\n---- SAVING DFN PLOT FOR ROTATION {idx+1} ({axis_name}-direction solve) ----")
                p1 = dfn.initiate_plotter(
                    title=True, off_screen=False, scale=1, axis=True)

                dfn.plot_fractures_head(
                    p1, 40, 10, opacity=1, contour=True
                )
                regbox.plot(p1)

                dfn_plotters[str(idx+1)] = p1

    # 3D plot where each point is a directional vector tip in local coordinates.
    if plot_points:
        points = np.array(plot_points)
        dirs = np.array(plot_dirs)
        k_meas = np.linalg.norm(points, axis=1)
        rot_ids = np.array(plot_labels)
        colors = np.array(plot_colors)
        face_ids = np.array(plot_face_ids)

        def fit_tensor_least_squares(direction_vectors, k_values):
            nx, ny, nz = (
                direction_vectors[:, 0], direction_vectors[:,
                                                           1], direction_vectors[:, 2],
            )
            design = np.column_stack(
                [nx**2, 2*nx*ny, 2*nx*nz, ny**2, 2*ny*nz, nz**2])
            coef, *_ = np.linalg.lstsq(design, k_values, rcond=None)
            k_tensor = np.array([
                [coef[0], coef[1], coef[2]],
                [coef[1], coef[3], coef[4]],
                [coef[2], coef[4], coef[5]],
            ])
            return k_tensor

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

        K_fit = fit_tensor_least_squares(dirs, k_meas)

        # ====== Export Rotations CSVs ======
        CSV_DIR = Path(r"C:\Users\SEVB94859\Flopy\flopythesis\csv_files")
        POINTS_CSV_PATH = CSV_DIR / "ny_rotation_points.csv"
        TENSOR_CSV_PATH = CSV_DIR / "ny_rotation_tensor_for_continuum.csv"

        CSV_DIR.mkdir(parents=True, exist_ok=True)

        header_points = "point_id,rotation_label,face,kx,ky,kz,k_value,dir_x,dir_y,dir_z\n"
        with open(POINTS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            f.write(header_points)
            for i, (p, d, r_label, face) in enumerate(zip(points, dirs, rot_ids, face_ids), start=1):
                k_val = float(np.linalg.norm(p))
                f.write(
                    f"{i},{r_label},{face},"
                    f"{p[0]:.10g},{p[1]:.10g},{p[2]:.10g},"
                    f"{k_val:.10g},{d[0]:.10g},{d[1]:.10g},{d[2]:.10g}\n"
                )

        header_tensor = "angle_deg,k_xx,k_xy,k_xz,k_yx,k_yy,k_yz,k_zx,k_zy,k_zz\n"
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

        # ====== Plot 1: Scatter Points ======
        # Re-use overlap y-points visualization
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
        colors_plot = colors[plot_mask]
        face_ids_plot = face_ids[plot_mask]
        rot_ids_plot = rot_ids[plot_mask]

        # Display offset to separate dots visually
        points_plot_display = points_plot.copy()
        face_axes_map = {
            "+x": np.array([1, 0, 0]), "-x": np.array([-1, 0, 0]),
            "+y": np.array([0, 1, 0]), "-y": np.array([0, -1, 0]),
            "+z": np.array([0, 0, 1]), "-z": np.array([0, 0, -1]),
        }
        for i, face in enumerate(face_ids_plot):
            points_plot_display[i] = points_plot_display[i] + \
                0.03 * face_axes_map[face]

        fig_scatter = plt.figure(figsize=(8, 7))
        ax_scatter = fig_scatter.add_subplot(111, projection="3d")

        ax_scatter.scatter(
            points_plot_display[:, 0], points_plot_display[:,
                                                           1], points_plot_display[:, 2],
            c=colors_plot, s=60, alpha=0.95, edgecolors="black", linewidths=0.35,
            label="K magnitude"
        )

        # Labels
        for p, r_label, face in zip(points_plot_display, rot_ids_plot, face_ids_plot):
            ax_scatter.text(p[0] + 0.03, p[1] + 0.03, p[2] +
                            0.03, f"{r_label}:{face}", fontsize=7, alpha=0.85)

        lim_sc = np.max(np.abs(points)) * 1.2
        ax_scatter.set_xlim(-lim_sc, lim_sc)
        ax_scatter.set_ylim(-lim_sc, lim_sc)
        ax_scatter.set_zlim(-lim_sc, lim_sc)
        ax_scatter.set_box_aspect([1, 1, 1])
        ax_scatter.set_xlabel("K_x")
        ax_scatter.set_ylabel("K_y")
        ax_scatter.set_zlabel("K_z")
        ax_scatter.set_title("Directional conductivity from flow experiments")
        plt.tight_layout()
        plt.show(block=False)

        # ====== Plot 2: Ellipsoid Fit ======
        fig_fit = plt.figure(figsize=(8, 7))
        ax_fit = fig_fit.add_subplot(111, projection="3d")

        ex, ey, ez, evals_fit, evecs_fit = build_ellipsoid_mesh_from_tensor(
            K_fit)

        ax_fit.plot_surface(ex, ey, ez, color="#8ecae6",
                            alpha=0.28, linewidth=0)

        ax_fit.scatter(
            points_plot[:, 0], points_plot[:, 1], points_plot[:, 2],
            c=colors_plot, s=40, alpha=0.8, edgecolors="black", linewidths=0.35,
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
        fig_fit.tight_layout()

        # Add a text box to the figure for picking a PyVista DFN plot to view
        fig_fit.subplots_adjust(bottom=0.2)
        ax_text = fig_fit.add_axes([0.35, 0.05, 0.3, 0.075])

        from matplotlib.widgets import TextBox
        plot_selector = TextBox(
            ax_text, "Visa 3D DFN plot nr (1-66): ", initial="")

        def submit_plot(text):
            text = text.strip()
            if not text:
                return
            if text in dfn_plotters:
                print(
                    f"Visar 3D DFN plot för rotation {text}... (Stäng fönstret för att fortsätta med Matplotlib)")
                try:
                    dfn_plotters[text].show()
                except Exception as e:
                    print(f"Fel vid visning (Kanske redan stängd?): {e}")
                plot_selector.set_val("")  # reset text box
            else:
                print(
                    f"Ingen plot hittades för '{text}'. (Värden t.ex. 1 till 66, eller failed solves)")
                plot_selector.set_val("")

        plot_selector.on_submit(submit_plot)

        if block_on_final_k_plot:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(0.1)

    end = datetime.datetime.now()
    print(f"\n\nProgram ended at {end}")
    print(f"Time elapsed: {end - start0}")
    print(f"\t-generating: \t{start1 - start0}")
    print(f"\t-solving: \t\t\t{start2 - start1}")
    print(f"\t-plotting: \t\t{end - start2}")

    if plot_points:
        eigvals, eigvecs = np.linalg.eigh(K_fit)
        order = np.argsort(eigvals)[::-1]

        print("\nFitted conductivity tensor K:")
        # Print with formatted 3x3 matrix specifically
        for row in K_fit:
            print(f"[{row[0]:>12.4e}, {row[1]:>12.4e}, {row[2]:>12.4e}]")

        print("\nPrincipal conductivities:")
        print(eigvals[order])

        print("\nPrincipal directions (columns):")
        print(eigvecs[:, order])

    print("\nAll done!")