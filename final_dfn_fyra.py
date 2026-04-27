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
from andfn import copy_dfn

# ======================================================
# Logger
# ======================================================

logger = logging.getLogger("dfn_run")

def setup_run_dir(base="runs"):
    """Create a timestamped output directory and attach a file logger to it."""
    run_dir = os.path.join(
        base, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    fh = logging.FileHandler(os.path.join(
        run_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    logger.info(f"Logging configured via setup_run_dir in {run_dir}")
    return run_dir


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

    save = True  # Sparar tensor_summary och slutgiltiga plots
    save_rotation_plots = True  # För att spara 3D-bilder för VARJE enskild rotation
    block_on_final_k_plot = True

    start0 = datetime.datetime.now()

    # Använd vår nya logger-funktion istället för att bygga mappen manuellt
    sys_dir_str = setup_run_dir(
        base=r"C:\Users\SEMB94861\Flopy\flopythesis\simulations")
    sim_dir = Path(sys_dir_str)

    plots_dir = sim_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Byt ut den gamla terminal-loggern mot the logger vi designat ovan

    logger.info(f"\nProgram started at {start0}")
    logger.info(f"Simulation output directory: {sim_dir}")

    # Eftersom resten av koden använder `print()`, kan vi slussa den in i the standard library loggern också om vi vill,
    # men vi behåller din Logger-klass tills vidare för att inte störa existerande kod-flöden i onödan.
    sys.stdout = Logger(sim_dir / "terminal_output.txt")

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
    # Rotation configurations
    # --------------------------------------------------
    rotation_configs = []

    # Initial rotation
    rotation_configs.append(([], ["x", "y", "z"], "start"))

    # Rotation around z-axies
    for z in range(15, 90, 15):
        rotation_configs.append(
            ([(z, (0, 0, 1))], ["x", "y"], f"base_z{z}")
        )
    # Tilting in y and the rotating around z again
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
    # Tilting in x and the rotating around z again
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

            # Vi kör två simuleringar per axel, en för negativ riktning och en för positiv.
            for direction_flow in ["neg", "pos"]:

                dfn = andfn.DFN("Copy", discharge_int=25)
                fracture_copy = copy_dfn(dfn_org.fractures)
                dfn.add_fracture(fracture_copy)

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

                # dfn = andfn.DFN("Rotated", discharge_int=25)
                # dfn.add_fracture(list(reg_fracs_in))

                if direction_flow == "neg":
                    # Flöde mot -x,y,z: Högre head på +x,y,z (face_high), lägre på -x,y,z (face_low)
                    regbox.frac_intersections(
                        dfn.fractures, face=face_low, head=head0)
                    regbox.frac_intersections(
                        dfn.fractures, face=face_high, head=head1)
                    target_face = faces[1]  # "-x", "-y" eller "-z"
                    target_vec = -v0
                else:
                    # Flöde mot +x,y,z: Högre head på -x,y,z (face_low), lägre på +x,y,z (face_high)
                    regbox.frac_intersections(
                        dfn.fractures, face=face_low, head=head1)
                    regbox.frac_intersections(
                        dfn.fractures, face=face_high, head=head0)
                    target_face = faces[0]  # "+x", "+y" eller "+z"
                    target_vec = v0

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
                        f"Solver failed for rot={rot_label}, axis={axis_name}, dir={direction_flow}: {exc}")

                start2 = datetime.datetime.now()

                if solve_failed:
                    sum_flows = np.nan
                    k_axis = np.nan
                else:
                    sum_flows = regbox.get_total_flow() / 2.0
                    k_axis = sum_flows * L / (A * (head1 - head0))

                if np.isnan(k_axis):
                    continue

                print(
                    f"Calculated k_{axis_name}n ({direction_flow}) för rotation {idx+1} ({rot_label}): {k_axis:.6e} m/s")

                n_rotated = rotate_vector(target_vec, rotations)

                plot_points.append(k_axis * n_rotated)
                plot_dirs.append(n_rotated)
                plot_colors.append(FACE_COLORS[target_face])
                plot_labels.append(rot_label)
                plot_face_ids.append(target_face)

                if not solve_failed:
                    if save_rotation_plots:
                        print(
                            f"\n---- SAVING DFN PLOT FOR ROTATION {idx+1} ({axis_name}-direction, {direction_flow}) ----")
                        p1 = dfn.initiate_plotter(
                            title=True, off_screen=True, scale=1, axis=True)

                        dfn.plot_fractures_head(
                            p1, 40, 10, opacity=1, contour=True
                        )
                        regbox.plot(p1)

                        img_path = plots_dir / \
                            f"dfn_plot_rot{idx+1}_{axis_name}_{direction_flow}.png"
                        html_path = plots_dir / \
                            f"dfn_plot_rot{idx+1}_{axis_name}_{direction_flow}.html"
                        # Save a screenshot of the PyVista plot
                        p1.screenshot(img_path)
                        p1.export_html(str(html_path))

                        if str(idx+1) not in dfn_plotters:
                            dfn_plotters[str(idx+1)] = {}
                        dfn_plotters[str(
                            idx+1)][f"{axis_name}_{direction_flow}"] = img_path
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

        K_fit, A, rmse_fit = fit_tensor_least_squares(dirs, k_meas)

        # Skapa en random 90/10 split för anpassning resp. validering
        num_pts = len(k_meas)
        indices = np.random.permutation(num_pts)
        val_size = max(1, int(0.1 * num_pts))
        val_idx = indices[:val_size]
        fit_idx = indices[val_size:]

        fit_dirs = dirs[fit_idx]
        fit_k = k_meas[fit_idx]
        val_dirs = dirs[val_idx]
        val_k = k_meas[val_idx]

        # Omkalkylera anpassningen medbart på träningspunkterna (90%)
        K_fit, A, rmse_fit = fit_tensor_least_squares(fit_dirs, fit_k)

        # Beräkna RMSE för valideringsdatan (hold-out)
        nx, ny, nz = val_dirs[:, 0], val_dirs[:, 1], val_dirs[:, 2]
        A_val = np.column_stack(
            [nx**2, 2*nx*ny, 2*nx*nz, ny**2, 2*ny*nz, nz**2])
        coef_fit = np.array(
            [K_fit[0, 0], K_fit[0, 1], K_fit[0, 2], K_fit[1, 1], K_fit[1, 2], K_fit[2, 2]])
        k_pred_val = A_val @ coef_fit
        rmse_val = float(np.sqrt(np.mean((val_k - k_pred_val) ** 2)))

        summary_lines = []

        def log_and_print(text):
            print(text)
            summary_lines.append(text)

        log_and_print(f"Rank(A): {np.linalg.matrix_rank(A)}")

        expected_points = 6 + 4 * 4 + 5 * \
            (4 + 6 * 5) + 5 * (4 + 4 * 5)  # Est. from original logic
        log_and_print(
            f"Total points: {len(points)} (expected ~{expected_points})")

        rot_order = list(dict.fromkeys(rot_ids.tolist()))
        for rot_label in rot_order:
            mask = rot_ids == rot_label
            unique_faces = sorted(set(face_ids[i] for i in np.where(mask)[0]))
            log_and_print(
                f"{rot_label}: {np.sum(mask)} pkt, faces={unique_faces}")

        log_and_print(
            f"RMSE residual (least squares, fit set): {rmse_fit:.6e} (0 = perfect ellipsoid)")

        Kxx, Kxy, Kxz = K_fit[0, 0], K_fit[0, 1], K_fit[0, 2]
        Kyy, Kyz = K_fit[1, 1], K_fit[1, 2]
        Kzz = K_fit[2, 2]

        evals_print, evecs_print = np.linalg.eigh(K_fit)
        order_idx = np.argsort(evals_print)[::-1]
        eigvals = evals_print[order_idx]
        eigvecs = evecs_print[:, order_idx]

        log_and_print("\nFitted 3D conductivity tensor")
        log_and_print(f"  K = [[{Kxx:.4e}, {Kxy:.4e}, {Kxz:.4e}],")
        log_and_print(f"       [{Kxy:.4e}, {Kyy:.4e}, {Kyz:.4e}],")
        log_and_print(f"       [{Kxz:.4e}, {Kyz:.4e}, {Kzz:.4e}]]")
        log_and_print(
            f"  Principal values: k1 = {eigvals[0]:.4e}, k2 = {eigvals[1]:.4e}, k3 = {eigvals[2]:.4e}")
        log_and_print("  Principal axes:")
        log_and_print(
            f"    v1 = [{eigvecs[0, 0]:.4f}, {eigvecs[1, 0]:.4f}, {eigvecs[2, 0]:.4f}]")
        log_and_print(
            f"    v2 = [{eigvecs[0, 1]:.4f}, {eigvecs[1, 1]:.4f}, {eigvecs[2, 1]:.4f}]")
        log_and_print(
            f"    v3 = [{eigvecs[0, 2]:.4f}, {eigvecs[1, 2]:.4f}, {eigvecs[2, 2]:.4f}]\n")

        for i in range(3):
            summary_lines.append(
                f"    v{i+1} = [{eigvecs[0, i]:.4f}, {eigvecs[1, i]:.4f}, {eigvecs[2, i]:.4f}]"
            )

        summary_lines += [
            "",
            f"  RMSE on fit directions:        {rmse_fit:.4e}",
            f"  RMSE on held-out (validation): {rmse_val:.4e}  <- key quality indicator",
        ]

        # --- SPD-constrained fit & Sanity Check ---
        import cvxpy as cp

        def build_design_matrix(direction_vectors):
            nx, ny, nz = direction_vectors[:,
                                           0], direction_vectors[:, 1], direction_vectors[:, 2]
            return np.column_stack([nx**2, 2*nx*ny, 2*nx*nz, ny**2, 2*ny*nz, nz**2])

        # Använd datan från vår 90/10 split
        fit_directions = fit_dirs
        val_directions = val_dirs
        cond = np.linalg.cond(A)

        # ---------------------------------------------------------------
        # SPD-constrained fit via cvxpy: solve the same least-squares but
        # require K >> 0 (positive semidefinite).
        # ---------------------------------------------------------------
        K_SCALE = 1.0 / max(np.max(np.abs(fit_k)), 1e-300)
        K_var = cp.Variable((3, 3), PSD=True)
        k_vec = cp.hstack([K_var[0, 0], K_var[0, 1], K_var[0, 2],
                           K_var[1, 1], K_var[1, 2], K_var[2, 2]])
        prob = cp.Problem(cp.Minimize(
            cp.sum_squares(A @ k_vec - fit_k * K_SCALE)))

        #  testar Clarabel, tillbaka på standard om den saknas
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            prob.solve()

        if K_var.value is None:
            # Fallback om CVXPY-solvern kraschar totalt
            K_spd = K_fit.copy()
        else:
            K_spd = np.array(K_var.value) / K_SCALE

        K_spd = 0.5 * (K_spd + K_spd.T)

        fit_k_pred_spd = build_design_matrix(fit_directions) @ np.array([
            K_spd[0, 0], K_spd[0, 1], K_spd[0, 2],
            K_spd[1, 1], K_spd[1, 2], K_spd[2, 2],
        ])
        val_k_pred_spd = build_design_matrix(val_directions) @ np.array([
            K_spd[0, 0], K_spd[0, 1], K_spd[0, 2],
            K_spd[1, 1], K_spd[1, 2], K_spd[2, 2],
        ])
        rmse_fit_spd = np.sqrt(np.mean((fit_k - fit_k_pred_spd) ** 2))
        rmse_val_spd = np.sqrt(np.mean((val_k - val_k_pred_spd) ** 2))

        eigvals_spd, eigvecs_spd = np.linalg.eigh(K_spd)
        order_spd = np.argsort(eigvals_spd)[::-1]
        eigvals_spd = eigvals_spd[order_spd]
        eigvecs_spd = eigvecs_spd[:, order_spd]

        summary_lines += [
            "",
            "SPD-constrained fit (K >> 0 via cvxpy):",
            f"  K_spd = [[{K_spd[0, 0]:.4e}, {K_spd[0, 1]:.4e}, {K_spd[0, 2]:.4e}],",
            f"           [{K_spd[0, 1]:.4e}, {K_spd[1, 1]:.4e}, {K_spd[1, 2]:.4e}],",
            f"           [{K_spd[0, 2]:.4e}, {K_spd[1, 2]:.4e}, {K_spd[2, 2]:.4e}]]",
            f"  Principal values (SPD): k1={eigvals_spd[0]:.4e}, k2={eigvals_spd[1]:.4e}, k3={eigvals_spd[2]:.4e}",
        ]

        for i in range(3):
            summary_lines.append(
                f"    v{i+1}_spd = [{eigvecs_spd[0, i]:.4f}, {eigvecs_spd[1, i]:.4f}, {eigvecs_spd[2, i]:.4f}]"
            )

        summary_lines += [
            f"  RMSE (fit, SPD):        {rmse_fit_spd:.4e}",
            f"  RMSE (validation, SPD): {rmse_val_spd:.4e}",
            f"  cvxpy status: {prob.status}",
        ]

        # ---------------------------------------------------------------
        # Axis-aligned sanity check
        # ---------------------------------------------------------------
        def _lookup_axis_k(directions, k_values, axis_vec, tol=1e-6):
            dots = np.abs(directions @ axis_vec)
            idx = int(np.argmax(dots))
            return k_values[idx] if dots[idx] > 1 - tol else np.nan

        k_axis = {
            "x": _lookup_axis_k(dirs, k_meas, np.array([1.0, 0, 0])),
            "y": _lookup_axis_k(dirs, k_meas, np.array([0, 1.0, 0])),
            "z": _lookup_axis_k(dirs, k_meas, np.array([0, 0, 1.0])),
        }

        summary_lines += [
            "",
            "Axis-aligned sanity check (direct DFN vs fitted diagonal):",
            f"  k_x (direct) = {k_axis['x']:.4e}   Kxx (lsq) = {Kxx:.4e}   Kxx (SPD) = {K_spd[0, 0]:.4e}",
            f"  k_y (direct) = {k_axis['y']:.4e}   Kyy (lsq) = {Kyy:.4e}   Kyy (SPD) = {K_spd[1, 1]:.4e}",
            f"  k_z (direct) = {k_axis['z']:.4e}   Kzz (lsq) = {Kzz:.4e}   Kzz (SPD) = {K_spd[2, 2]:.4e}",
            "  (Large |direct - fit| along an axis indicates the tensor is",
            "   underconstrained in that direction -- e.g. near-2D fracture set.)",
        ]

        # ---------------------------------------------------------------
        # Paper-style normalized relative error Er (Wang et al. 2023)
        # ---------------------------------------------------------------
        def _paper_k_pred(directions, K_mat):
            return np.einsum("ij,jk,ik->i", directions, K_mat, directions)

        def _paper_er(k_meas_arr, k_pred):
            good = np.isfinite(k_pred) & (k_pred > 0) & (k_meas_arr > 0)
            if not np.any(good):
                return np.nan
            return float(np.mean(np.abs(1.0 - k_meas_arr[good] / k_pred[good])))

        k_pred_lsq_all = _paper_k_pred(dirs, K_fit)
        k_pred_spd_all = _paper_k_pred(dirs, K_spd)
        er_lsq = _paper_er(k_meas, k_pred_lsq_all)
        er_spd = _paper_er(k_meas, k_pred_spd_all)
        er_lsq_val = _paper_er(val_k, _paper_k_pred(val_directions, K_fit))
        er_spd_val = _paper_er(val_k, _paper_k_pred(val_directions, K_spd))

        summary_lines += [
            "",
            "Paper-style normalized relative error Er (Wang et al. 2023, eq. 9):",
            f"  Er (lsq fit, all directions):  {er_lsq:.4f}",
            f"  Er (lsq fit, validation only): {er_lsq_val:.4f}",
            f"  Er (SPD fit, all directions):  {er_spd:.4f}",
            f"  Er (SPD fit, validation only): {er_spd_val:.4f}",
            "  (Er < 0.3 is the paper's KREV acceptance threshold.)",
        ]

        # Save summary to file
        summary_path = sim_dir / "tensor_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print(f"Saved tensor summary to: {summary_path}")

        # Spara tensordatan
        try:
            np.savez(sim_dir / "tensor.npz",
                     K=K_fit, eigvals=eigvals, eigvecs=eigvecs,
                     rmse_fit=rmse_fit, rmse_val=rmse_val, cond=cond,
                     K_spd=K_spd, eigvals_spd=eigvals_spd, eigvecs_spd=eigvecs_spd,
                     rmse_fit_spd=rmse_fit_spd, rmse_val_spd=rmse_val_spd,
                     k_axis_x=k_axis["x"], k_axis_y=k_axis["y"], k_axis_z=k_axis["z"],
                     er_lsq=er_lsq, er_spd=er_spd,
                     er_lsq_val=er_lsq_val, er_spd_val=er_spd_val,
                     fit_directions=fit_directions, fit_k=fit_k,
                     val_directions=val_directions, val_k=val_k,
                     fit_idx=fit_idx, val_idx=val_idx)
            print(f"Saved tensor.npz to: {sim_dir / 'tensor.npz'}")
        except Exception as e:
            print(f"Kunde inte spara tensor.npz: {e}")

        # ====== Export Rotations CSVs ======
        # Save only in simulations folder
        POINTS_CSV_PATH = sim_dir / "ny_rotation_points.csv"
        TENSOR_CSV_PATH = sim_dir / "ny_rotation_tensor_for_continuum.csv"

        header_points = "point_id;rotation_label;face;kx;ky;kz;k_value;dir_x;dir_y;dir_z\n"
        with open(POINTS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            f.write(header_points)
            for i, (p, d, r_label, face) in enumerate(zip(points, dirs, rot_ids, face_ids), start=1):
                k_val = float(np.linalg.norm(p))
                f.write(
                    f"{i};{r_label};{face};"
                    f"{p[0]:.10g};{p[1]:.10g};{p[2]:.10g};"
                    f"{k_val:.10g};{d[0]:.10g};{d[1]:.10g};{d[2]:.10g}\n"
                )

        header_tensor = "angle_deg;k_xx;k_xy;k_xz;k_yx;k_yy;k_yz;k_zx;k_zy;k_zz\n"
        with open(TENSOR_CSV_PATH, "w", encoding="utf-8", newline="") as f:
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
        if save:
            fig_fit.savefig(plots_dir / "fitted_ellipsoid.png", dpi=300)

            # Exportera den anpassade ellipsoiden (fitted ellipsoid) som VTK och HTML via PyVista
            import pyvista as pv
            grid = pv.StructuredGrid(ex, ey, ez)
            grid.save(str(plots_dir / "fitted_ellipsoid.vtk"))

            pl_fit = pv.Plotter(off_screen=True)
            pl_fit.add_mesh(grid, color="#8ecae6", opacity=0.3)

            # Lägg till våra mätpunkter
            point_cloud_fit = pv.PolyData(points)
            pl_fit.add_points(point_cloud_fit, color="red", point_size=8)

            # Spara som interaktiv HTML
            pl_fit.export_html(str(plots_dir / "fitted_ellipsoid.html"))
            pl_fit.close()

        # ====== Plot 2: Image Viewer for DFN Plots ======
        if dfn_plotters:
            fig_viewer = plt.figure(figsize=(9, 8))
            fig_viewer.subplots_adjust(bottom=0.2, top=0.95)
            ax_viewer = fig_viewer.add_subplot(111)

            from matplotlib.widgets import TextBox, RadioButtons
            import matplotlib.image as mpimg

            state = {"sim": next(iter(dfn_plotters.keys())), "axis": "x_neg"}

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

            ax_radio = fig_viewer.add_axes([0.05, 0.7, 0.15, 0.25])
            radio = RadioButtons(
                ax_radio, ('x_neg', 'x_pos', 'y_neg', 'y_pos', 'z_neg', 'z_pos'), active=0)

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

        # --- Additional Advanced 3D Plots (Shell, scatter etc.) ---
        import pyvista as pv
        from scipy.spatial import ConvexHull

        all_points = points
        fit_points = points

        # --- 3D scatter plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(fit_points[:, 0], fit_points[:, 1], fit_points[:, 2],
                   c="red", marker="o", s=10, label="Fit measurements")
        ax.set_xlabel("Conductivity x")
        ax.set_ylabel("Conductivity y")
        ax.set_zlabel("Conductivity z")
        ax.set_title("3D Effective Conductivity - Rotation Grid Sampling")
        ax.legend()
        for azim in (30, 120, 210, 300):
            ax.view_init(elev=20, azim=azim)
            fig.savefig(plots_dir / f"scatter_3d_azim{azim:03d}.png",
                        dpi=200, bbox_inches="tight")
        ax.view_init(elev=20, azim=30)
        plt.show(block=False)

        # --- Paper-style hydraulic conductivity ellipsoid (Wang et al. 2023) ---
        sphere = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
        sphere_pts = np.array(sphere.points)

        eigvals_plot = np.clip(eigvals_spd, 0.0, None)
        paper_pts = (sphere_pts * np.sqrt(eigvals_plot)) @ eigvecs_spd.T
        paper_ellipsoid = sphere.copy()
        paper_ellipsoid.points = paper_pts
        paper_ellipsoid.save(str(plots_dir / "paper_ellipsoid.vtk"))

        sqrt_fit_points = fit_directions * np.sqrt(fit_k)[:, None]
        sqrt_val_points = val_directions * np.sqrt(val_k)[:, None]

        plotter_paper = pv.Plotter(
            title="Paper-style conductivity ellipsoid (sqrt(k))", off_screen=False)
        plotter_paper.add_mesh(paper_ellipsoid, color="lightgreen", opacity=0.4,
                               label="sqrt(K) ellipsoid (Wang et al. 2023)")
        plotter_paper.add_points(sqrt_fit_points, color="red", point_size=8,
                                 render_points_as_spheres=True,
                                 label="sqrt(k_meas) * n (fit)")
        plotter_paper.add_points(sqrt_val_points, color="orange", point_size=12,
                                 render_points_as_spheres=True,
                                 label="sqrt(k_meas) * n (validation)")
        paper_labels = ["sqrt(K1) (max)", "sqrt(K2) (mid)", "sqrt(K3) (min)"]
        for i, color in enumerate(["red", "green", "blue"]):
            semi = eigvecs_spd[:, i] * np.sqrt(eigvals_plot[i])
            plotter_paper.add_arrows(np.zeros((1, 3)), semi.reshape(1, 3),
                                     color=color, mag=1)
            plotter_paper.add_point_labels(
                semi.reshape(1, 3),
                [f"{paper_labels[i]} = {np.sqrt(eigvals_plot[i]):.4e}"],
                font_size=12, text_color=color, bold=True,
            )
        plotter_paper.add_text(
            f"Er (SPD, all) = {er_spd:.3f}   Er (SPD, val) = {er_spd_val:.3f}\n"
            f"(Er < 0.3 -> KREV acceptance)",
            font_size=10, position="upper_left",
        )
        plotter_paper.add_legend()
        plotter_paper.add_axes()
        plotter_paper.camera_position = "iso"
        plotter_paper.export_html(str(plots_dir / "paper_ellipsoid.html"))
        plotter_paper.show(screenshot=str(plots_dir / "paper_ellipsoid.png"),
                           auto_close=False, interactive_update=True)

        # --- Triangulated shell surface of the measured ellipsoid
        if len(all_points) >= 4:
            shell_norms = np.linalg.norm(all_points, axis=1, keepdims=True)
            shell_unit = all_points / \
                np.where(shell_norms > 0, shell_norms, 1.0)
            shell_hull = ConvexHull(shell_unit)
            shell_faces = np.hstack([
                np.full((len(shell_hull.simplices), 1), 3, dtype=np.intp),
                shell_hull.simplices,
            ]).ravel()
            shell_mesh = pv.PolyData(all_points, shell_faces)

            point_cloud = pv.PolyData(all_points)
            point_cloud["point_id"] = np.arange(
                1, len(all_points) + 1, dtype=int)

            plotter_shell = pv.Plotter(
                title="Measured Ellipsoid Shell", off_screen=False)
            plotter_shell.add_mesh(shell_mesh, color="lightsteelblue",
                                   opacity=0.30, show_edges=True)
            plotter_shell.add_points(point_cloud, color="darkred", point_size=10,
                                     label="Measurement points")
            plotter_shell.add_axes()
            plotter_shell.add_title(
                f"Measured Ellipsoid - Triangulated Shell\n({len(all_points)} points)"
            )
            plotter_shell.camera_position = "iso"
            plotter_shell.show(screenshot=str(plots_dir / "shell.png"),
                               auto_close=False, interactive_update=True)
            plotter_shell.export_html(str(plots_dir / "shell.html"))
            shell_mesh.save(str(plots_dir / "shell.vtk"))

        if block_on_final_k_plot:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(0.1)

    end = datetime.datetime.now()
    print(f"\nProgram ended at {end}")
    print(f"Time elapsed: {end - start0}")
