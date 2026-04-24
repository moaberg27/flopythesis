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

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
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

    save = True
    block_on_final_k_plot = True

    start0 = datetime.datetime.now()

    sim_dir = Path(
        r"C:\Users\SEMB94861\Flopy\flopythesis\simulations"
    ) / start0.strftime("%Y%m%d_%H%M%S")
    sim_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = Logger(sim_dir / "terminal_output.txt")

    print(f"\nProgram started at {start0}")
    print(f"Simulation output directory: {sim_dir}")

    # --------------------------------------------------
    # Load DFN
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Rotation configurations (NY STRUKTUR – SAMMA INNEBÖRD)
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
        "+y": "#005eff", "-y": "#0b98f0",
        "+z": "#2a9d8f", "-z": "#1b7f6b",
    }

    axis_faces = {
        "x": ("left", "right"),
        "y": ("front", "back"),
        "z": ("top", "bottom"),
    }

    head0, head1 = 100, 200

    dfn_plotters = {}

    # --------------------------------------------------
    # Main simulation loop
    # --------------------------------------------------
    for idx, (rotations, active_axes, rot_label) in enumerate(rotation_configs):

        print(f"\n[{idx+1}/{len(rotation_configs)}] === Rotation {rot_label} ===")

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
            dfn.set_kwargs(
                COEF_RATIO=0.001,
                MAX_ITERATIONS=30,
                MAX_NCOEF=200,
                MAX_ERROR=5e-4,
            )

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
                    regbox.plot(p1)

                    img_path = sim_dir / f"dfn_plot_rot{idx+1}_{axis_name}.png"
                    # Save a screenshot of the PyVista plot
                    p1.screenshot(img_path)

                    if str(idx+1) not in dfn_plotters:
                        dfn_plotters[str(idx+1)] = {}
                    dfn_plotters[str(idx+1)][axis_name] = img_path
                    p1.close()

    # --------------------------------------------------
    # ===== PLOTTING & EXPORT =====
    # --------------------------------------------------
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
        # Keep copy in original csv_files folder and simulations folder
        CSV_DIR_ORIG = Path(r"C:\Users\SEMB94861\Flopy\flopythesis\csv_files")

        for CSV_DIR in [sim_dir, CSV_DIR_ORIG]:
            CSV_DIR.mkdir(parents=True, exist_ok=True)
            POINTS_CSV_PATH = CSV_DIR / "ny_rotation_points.csv"
            TENSOR_CSV_PATH = CSV_DIR / "ny_rotation_tensor_for_continuum.csv"

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
        if save:
            fig_fit.savefig(sim_dir / "fitted_ellipsoid.png", dpi=300)

        # ====== Plot 2: Image Viewer for DFN Plots ======
        if dfn_plotters:
            fig_viewer = plt.figure(figsize=(9, 8))
            fig_viewer.subplots_adjust(bottom=0.2, top=0.95)
            ax_viewer = fig_viewer.add_subplot(111)

            from matplotlib.widgets import TextBox, RadioButtons
            import matplotlib.image as mpimg

            state = {"sim": next(iter(dfn_plotters.keys())), "axis": "x"}

            def draw_image():
                sim = state["sim"]
                axis = state["axis"]
                if sim in dfn_plotters and axis in dfn_plotters[sim]:
                    try:
                        new_img = mpimg.imread(dfn_plotters[sim][axis])
                        ax_viewer.clear()
                        ax_viewer.imshow(new_img)
                        ax_viewer.axis("off")
                        ax_viewer.set_title(
                            f"DFN Plot: Simulering {sim}, Gradient: {axis}")
                        fig_viewer.canvas.draw_idle()
                    except Exception as e:
                        print(f"Fel vid inläsning av bild: {e}")
                else:
                    ax_viewer.clear()
                    ax_viewer.text(0.5, 0.5, f"Ingen plot tillgänglig för\nsimulering {sim} axel {axis}",
                                   ha='center', va='center', fontsize=12)
                    ax_viewer.axis("off")
                    ax_viewer.set_title(
                        f"DFN Plot: Simulering {sim}, Gradient: {axis}")
                    fig_viewer.canvas.draw_idle()

            ax_radio = fig_viewer.add_axes([0.05, 0.8, 0.1, 0.15])
            radio = RadioButtons(ax_radio, ('x', 'y', 'z'), active=0)

            def on_radio_click(label):
                state["axis"] = label
                draw_image()

            radio.on_clicked(on_radio_click)

            draw_image()

            ax_text = fig_viewer.add_axes([0.35, 0.05, 0.3, 0.075])
            plot_selector = TextBox(
                ax_text, "Visa 3D DFN plot nr (1-66): ", initial=state["sim"])

            def submit_plot(text):
                text = text.strip()
                if not text:
                    return
                if text in dfn_plotters:
                    state["sim"] = text
                    draw_image()
                else:
                    print(f"Ingen plot hittades för '{text}'.")

            plot_selector.on_submit(submit_plot)

        if block_on_final_k_plot:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(0.1)

    end = datetime.datetime.now()
    print(f"\nProgram ended at {end}")
    print(f"Time elapsed: {end - start0}")
