"""
DFN flow solver driven by the rotation scheme from rotation.py.
 
For each unique rotation state on a phi/theta grid, a fresh copy of the
reference DFN is built, the region box is rotated accordingly, constant-head
BCs are imposed on the selected face pair, and the equivalent conductivity
is computed from the total flow.  The collected (direction, k) pairs are
then used to fit the symmetric 3-D conductivity tensor.
"""
 
import csv
import datetime
import logging
import os
 
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull
 
import andfn
from andfn import copy_dfn
 
logger = logging.getLogger("dfn_run")
 
 
def setup_run_dir(base="runs"):
    """Create a timestamped output directory and attach a file logger to it."""
    run_dir = os.path.join(base, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
 
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    fh = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return run_dir
 
 
# =====================================================================
# ROTATION HELPERS (adapted from rotation.py)
# =====================================================================
FACE_NORMALS_LOCAL = np.array([
    [1, 0, 0],   # face_index 0 -> local X
    [0, 1, 0],   # face_index 1 -> local Y
    [0, 0, 1],   # face_index 2 -> local Z
], dtype=float)
 
# andfn face names for each local axis (low-head face, high-head face)
FACE_NAME_PAIRS = {
    0: ("left",   "right"),   # y-z plane, normal along X
    1: ("front",  "back"),    # x-z plane, normal along Y
    2: ("bottom", "top"),     # x-y plane, normal along Z
}
 
 
def rotation_matrix(axis, angle_deg):
    """Return the 3x3 rotation matrix for rotating `angle_deg` around `axis`."""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0,  0],
                         [0, c, -s],
                         [0, s,  c]])
    if axis == 'z':
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]])
    raise ValueError(f"axis must be 'x' or 'z', got {axis!r}")
 
 
def build_design_matrix(directions):
    """A @ [Kxx, Kxy, Kxz, Kyy, Kyz, Kzz] = k_meas."""
    nx, ny, nz = directions[:, 0], directions[:, 1], directions[:, 2]
    return np.column_stack([
        nx ** 2,
        2 * nx * ny,
        2 * nx * nz,
        ny ** 2,
        2 * ny * nz,
        nz ** 2,
    ])
 
 
def precompute_directions(rotations):
    """Unique lab-frame measurement directions from the phi/theta grid.
 
    Deduplicates so k(n) = k(-n) is not resampled.  Returns directions and
    the (phi_deg, theta_deg, face_index) state needed to realise each one
    physically inside the DFN.
    """
    step = rotations[1] - rotations[0]
    phi_angles   = list(range(0, 360, step))
    theta_angles = list(range(0, 180, step))
 
    seen = set()
    directions = []
    states = []
    for phi_deg in phi_angles:
        for theta_deg in theta_angles:
            R = rotation_matrix('x', theta_deg) @ rotation_matrix('z', phi_deg)
            for face_idx, n_local in enumerate(FACE_NORMALS_LOCAL):
                n_lab = R @ n_local
                key = tuple(np.round(n_lab, 8))
                if key not in seen:
                    seen.add(key)
                    directions.append(n_lab)
                    states.append((phi_deg, theta_deg, face_idx))
    return np.array(directions), states
 
 
def split_indices(n_total, val_fraction=0.2, seed=42):
    """Random fit / validation index split."""
    n_val = max(1, int(n_total * val_fraction))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    return idx[:-n_val], idx[-n_val:]
 
 
# =====================================================================
# DFN SIMULATION DRIVER
# =====================================================================
BOX_CENTER = [0.0, 0.0, 0.0]
BOX_SIZE   = 500.0       # cubic domain edge length (m)
HEAD_LOW   = 100.0
HEAD_HIGH  = 200.0
 
 
def _save_head_plot(dfn, regbox, out_path_stem):
    """Render the fracture head field off-screen and export it as an
    interactive PyVista plot (HTML) plus the underlying mesh (VTK)."""
    try:
        p = dfn.initiate_plotter(title=True, off_screen=True, scale=1, axis=True)
        dfn.plot_fractures_head(p, 40, 10, opacity=1, contour=True)
        regbox.plot(p)
        p.export_html(out_path_stem + ".html")
        try:
            p.export_vtkjs(out_path_stem + ".vtkjs")
        except Exception:
            pass
        p.close()
    except Exception as exc:
        logger.warning(f"head plot save failed for {out_path_stem}: "
                       f"{type(exc).__name__}: {exc}")
 
 
# Convergence tolerance for the inlet/outlet flow-balance check. At a truly
# converged steady-state solve the sum of |q| on the inlet face equals the
# sum on the outlet face. If they disagree by more than this fraction the
# run is treated as diverged.
FLOW_IMBALANCE_TOL = 0.05  # 5%
 
 
def _flow_by_head(regbox, head_value, tol=1e-9):
    """Sum |q| over ConstantHeadLine elements at a given BC head value."""
    from andfn.const_head import ConstantHeadLine
    return float(np.sum([
        np.abs(e.q) for e in regbox.elements
        if isinstance(e, ConstantHeadLine) and abs(e.head - head_value) < tol
    ]))
 
 
def _solver_converged(regbox):
    """Return (ok, imbalance, Q_low, Q_high).
 
    A diverged solver produces grossly different flow on the two BC faces
    (mass not conserved) -- the cleanest signal we can get without poking
    at andfn's iteration residuals.
    """
    Q_low  = _flow_by_head(regbox, HEAD_LOW)
    Q_high = _flow_by_head(regbox, HEAD_HIGH)
    q_max = max(Q_low, Q_high)
    if q_max == 0.0:
        return False, np.inf, Q_low, Q_high
    imbalance = abs(Q_low - Q_high) / q_max
    return imbalance < FLOW_IMBALANCE_TOL, imbalance, Q_low, Q_high
 
 
def _nan_pt_metrics():
    """Return a dict of porosity / travel-time metrics filled with NaN."""
    return {
        "phi_geo_all":     np.nan,
        "phi_geo_flow":    np.nan,
        "phi_eff_mean":    np.nan,
        "phi_eff_fast":    np.nan,
        "phi_eff_slow":    np.nan,
        "tau_geo_all_yr":  np.nan,
        "tau_geo_flow_yr": np.nan,
        "tau_mean_yr":     np.nan,
        "tau_fast_yr":     np.nan,
        "tau_slow_yr":     np.nan,
        "n_streamlines":   0,
        "_time_list_s":    [],
        "_length_list_m":  [],
    }


def _run_particle_tracking(dfn, regbox):
    """Particle tracking on an already-solved DFN; returns porosity metrics dict.

    Uses an off-screen PyVista plotter (required by plot_streamline_tracking)
    that is closed immediately after tracking -- no screenshots are saved per run.
    """
    from andfn.const_head import ConstantHeadLine as _CHL

    V_bulk = regbox.xl * regbox.yl * regbox.zl
    Q_low  = _flow_by_head(regbox, HEAD_LOW)
    Q_high = _flow_by_head(regbox, HEAD_HIGH)
    Q_tot  = float(0.5 * (Q_low + Q_high))

    phi_geo_all = sum(
        np.pi * f.radius ** 2 * (f.aperture if f.aperture is not None else 0.0)
        for f in dfn.fractures
    ) / V_bulk

    flowing_frac_ids = {
        id(e.frac0)
        for e in dfn.elements
        if isinstance(e, _CHL) and e.q < -1e-16
    }
    phi_geo_flow = sum(
        np.pi * f.radius ** 2 * (f.aperture if f.aperture is not None else 0.0)
        for f in dfn.fractures
        if id(f) in flowing_frac_ids
    ) / V_bulk

    pt_plotter = dfn.initiate_plotter(off_screen=True)
    cnt = 0
    ss, vel, el = [], [], []
    for e in dfn.elements:
        if isinstance(e, _CHL) and e.q < -1e-16:
            z_start = e.z_array_tracking(3, offset=1e-1)
            elevation = [0.125, 0.25, 0.5, 0.75, 0.875]
            streamlines, _, velocities, elements = dfn.plot_streamline_tracking(
                pt_plotter, z_start, e.frac0,
                ds=1e-2, max_length=5e3, line_width=4,
                elevation=elevation, remove_false=True, backward=False,
            )
            ss.append(streamlines)
            vel.append(velocities)
            el.append(elements)
            cnt += 1
            if cnt > 100:
                break
    pt_plotter.close()

    time_list = []
    for strem, velo, elem in zip(ss, vel, el):
        for v, s, ee in zip(velo, strem, elem):
            if ee is False or len(s) == 0:
                continue
            t, _ = dfn.get_travel_time_and_length(s, v)
            time_list.append(t)

    length_list = []
    for strem, velo, elem in zip(ss, vel, el):
        for v, s, ee in zip(velo, strem, elem):
            if ee is False or len(s) == 0:
                continue
            _, l = dfn.get_travel_time_and_length(s, v)
            length_list.append(l)

    _yr = 3600.0 * 24.0 * 365.0
    tau_geo_all_yr  = (phi_geo_all  * V_bulk / Q_tot) / _yr if Q_tot > 0 else np.nan
    tau_geo_flow_yr = (phi_geo_flow * V_bulk / Q_tot) / _yr if Q_tot > 0 else np.nan

    if len(time_list) > 0 and Q_tot > 0:
        tau_mean_s = float(np.mean(time_list))
        tau_fast_s = float(np.min(time_list))
        tau_slow_s = float(np.max(time_list))
        phi_eff_mean = (tau_mean_s * Q_tot) / V_bulk
        phi_eff_fast = (tau_fast_s * Q_tot) / V_bulk
        phi_eff_slow = (tau_slow_s * Q_tot) / V_bulk
        tau_mean_yr  = tau_mean_s / _yr
        tau_fast_yr  = tau_fast_s / _yr
        tau_slow_yr  = tau_slow_s / _yr
    else:
        phi_eff_mean = phi_eff_fast = phi_eff_slow = np.nan
        tau_mean_yr  = tau_fast_yr  = tau_slow_yr  = np.nan

    return {
        "phi_geo_all":     phi_geo_all,
        "phi_geo_flow":    phi_geo_flow,
        "phi_eff_mean":    phi_eff_mean,
        "phi_eff_fast":    phi_eff_fast,
        "phi_eff_slow":    phi_eff_slow,
        "tau_geo_all_yr":  tau_geo_all_yr,
        "tau_geo_flow_yr": tau_geo_flow_yr,
        "tau_mean_yr":     tau_mean_yr,
        "tau_fast_yr":     tau_fast_yr,
        "tau_slow_yr":     tau_slow_yr,
        "n_streamlines":   cnt,
        "_time_list_s":    time_list,       # raw travel times in seconds
        "_length_list_m":  length_list,     # raw trace lengths in metres
    }


def run_dfn_simulation(dfn_org, phi_deg, theta_deg, face_index):
    """Run one DFN flow simulation and return (k_eff, pt_metrics).
 
    The region box is rotated by phi around Z and then theta around X
    (matching R = Rx(theta) @ Rz(phi) applied to local face normals).
    Constant-head BCs are imposed on the face pair selected by
    `face_index`: 0 -> left/right (local X), 1 -> front/back (local Y),
    2 -> bottom/top (local Z).
 
    Returns (np.nan, _nan_pt_metrics()) if the solver fails so the main loop
    can continue. Also returns NaN k when the solver runs to MAX_ITERATIONS
    without the inlet/outlet flows balancing (detected via `_solver_converged`);
    oblique rotations sometimes produce near-degenerate intersections
    that the solver cannot resolve, and the resulting k values are
    meaningless and would poison the tensor fit.

    On a diverged first pass the solve is retried once with more
    iterations and more coefficients; only a persistently unbalanced
    run is rejected.

    `copy_dfn` gives this run its own Fracture objects, so mutations from
    `delete_fracture` / `frac_intersections` / `solve` don't leak back into
    `dfn_org` and successive runs start from the pristine imported state
    automatically.
    """
    try:
        dfn = andfn.DFN("Copy", discharge_int=50)
        fracture_copy = copy_dfn(dfn_org.fractures)
        dfn.add_fracture(fracture_copy)
 
        regbox = andfn.RectangularRegion(
            label="box",
            center=BOX_CENTER,
            x_vec=[1, 0, 0],
            y_vec=[0, 1, 0],
            z_vec=[0, 0, 1],
            xl=BOX_SIZE, yl=BOX_SIZE, zl=BOX_SIZE,
        )
        # andfn.rotate is extrinsic (lab-frame axis), so Z first then X reproduces
        # R = Rx(theta) @ Rz(phi) acting on the local face normals.
        if phi_deg:
            regbox.rotate(angle=phi_deg, axis=[0, 0, 1])
        if theta_deg:
            regbox.rotate(angle=theta_deg, axis=[1, 0, 0])
 
        _, frac_out = regbox.check_fractures(dfn.fractures, tree=dfn.tree)
        dfn.delete_fracture(frac_out)
 
        low_face, high_face = FACE_NAME_PAIRS[face_index]
        regbox.frac_intersections(dfn.fractures, face=low_face,  head=HEAD_LOW)
        regbox.frac_intersections(dfn.fractures, face=high_face, head=HEAD_HIGH)
 
        dfn.check_connectivity()
        dfn.set_kwargs(COEF_RATIO=0.001, MAX_ITERATIONS=30,
                       MAX_NCOEF=50, MAX_ERROR=5e-3)
        # Size the coefficient/integration arrays on the freshly-copied
        # elements before solving.
        dfn.reset(ncoef=5, nint=10)
        dfn.solve(unconsolidate=True)
 
        ok, imbalance, Q_low, Q_high = _solver_converged(regbox)
        if not ok:
            logger.warning(
                f"  diverged (phi={phi_deg}, theta={theta_deg}, face={face_index}): "
                f"Q_low={Q_low:.3e}, Q_high={Q_high:.3e}, "
                f"imbalance={imbalance:.2%}; retrying with more budget"
            )
            # Clear solver state and retry with more iterations / coefficients.
            # The geometry (fracture lists, BC elements) is unchanged -- only
            # the coefficients and discharge factorization are reset.
            dfn.reset(ncoef=5, nint=10)
            dfn.set_kwargs(COEF_RATIO=0.001, MAX_ITERATIONS=100,
                           MAX_NCOEF=100, MAX_ERROR=5e-3)
            dfn.solve(unconsolidate=True)
            ok, imbalance, Q_low, Q_high = _solver_converged(regbox)
            if not ok:
                logger.warning(
                    f"  still diverged after retry "
                    f"(imbalance={imbalance:.2%}); rejecting"
                )
                return np.nan, _nan_pt_metrics()
 
        Q  = 0.5 * (Q_low + Q_high)  # averaged inlet/outlet flow
        dH = HEAD_HIGH - HEAD_LOW
        L  = BOX_SIZE
        A  = BOX_SIZE * BOX_SIZE
        k_eff = Q * L / (A * dH) # effective conductivity from Darcy's law: Q = k A dH / L

        try:
            pt = _run_particle_tracking(dfn, regbox)
        except Exception as pt_exc:
            import traceback as _tb
            logger.warning(
                f"particle tracking failed "
                f"(phi={phi_deg}, theta={theta_deg}, face={face_index}): "
                f"{type(pt_exc).__name__}: {pt_exc}\n{_tb.format_exc()}"
            )
            pt = _nan_pt_metrics()

        return k_eff, pt
    except Exception as exc:
        import traceback
        logger.warning(
            f"simulation failed "
            f"(phi={phi_deg}, theta={theta_deg}, face={face_index}): "
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )
        return np.nan, _nan_pt_metrics()
 
 
# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    save = False
    scale = 1
    tracking = True
    animate = False
 
    ncoef = 10
    nint  = ncoef * 2
 
    run_dir = setup_run_dir()
    start0 = datetime.datetime.now()
    logger.info("---- IMPORT DFN ----")
    logger.info(f"Program started at {start0}")
    logger.info(f"Output directory: {run_dir}")
 
    dfn_org = andfn.DFN("DFN test FracMan", discharge_int=50)
    path = os.path.join(r"C:\Users\SEAM94860\FLOPY\finalflopy\flopythesis\dfn_6.csv")
    logger.info(f"DFN importing from file: {path}")
    dfn_org.import_fractures_from_file(
        path,
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
 
    # ---------------------------------------------------------------
    # PHASE 1 - precompute unique measurement directions (no DFN cost)
    # ---------------------------------------------------------------
    rotations = [0, 15, 30, 45, 60, 75]
    all_directions, all_rotation_states = precompute_directions(rotations)
 
    logger.info(f"Rotation grid step: {rotations[1] - rotations[0]} deg")
    logger.info(f"Unique measurement directions: {len(all_directions)}")
    logger.info("  -> one DFN simulation per direction")
 
    with open(os.path.join(run_dir, "unique_directions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nx", "ny", "nz"])
        w.writerows(all_directions)
    with open(os.path.join(run_dir, "rotation_states.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi_deg", "theta_deg", "face_index"])
        w.writerows(all_rotation_states)
    logger.info("Exported 'unique_directions.csv' and 'rotation_states.csv'")
 
    # ---------------------------------------------------------------
    # PHASE 2 - run DFN for every unique direction
    # ---------------------------------------------------------------
    start1 = datetime.datetime.now()
    logger.info("---- SOLVE THE DFN FOR EACH DIRECTION ----")
 
    n_total = len(all_rotation_states)
    all_k = np.empty(n_total)
    results_path = os.path.join(run_dir, "simulation_results.csv")
    all_times_s   = []   # aggregated travel times (seconds) across all rotation states
    all_lengths_m = []   # aggregated trace lengths (m)   across all rotation states
    with open(results_path, "w", newline="") as rf:
        rw = csv.writer(rf)
        rw.writerow([
            "i", "phi_deg", "theta_deg", "face",
            "nx", "ny", "nz", "k_eff", "status",
            "phi_geo_all", "phi_geo_flow",
            "phi_eff_mean", "phi_eff_fast", "phi_eff_slow",
            "tau_geo_all_yr", "tau_geo_flow_yr",
            "tau_mean_yr", "tau_fast_yr", "tau_slow_yr",
            "n_streamlines",
        ])
        for i, state in enumerate(all_rotation_states):
            logger.info(f"start run  {i + 1:>3}/{n_total}  "
                        f"phi={state[0]:>3} deg  theta={state[1]:>3} deg  "
                        f"face={state[2]}")
            k, pt = run_dfn_simulation(dfn_org, *state)
            all_k[i] = k
            nx, ny, nz = all_directions[i]
            status = "ok" if (np.isfinite(k) and k > 0) else "invalid"
            rw.writerow([
                i, state[0], state[1], state[2],
                f"{nx:.8f}", f"{ny:.8f}", f"{nz:.8f}",
                f"{k:.6e}", status,
                pt["phi_geo_all"], pt["phi_geo_flow"],
                pt["phi_eff_mean"], pt["phi_eff_fast"], pt["phi_eff_slow"],
                pt["tau_geo_all_yr"], pt["tau_geo_flow_yr"],
                pt["tau_mean_yr"], pt["tau_fast_yr"], pt["tau_slow_yr"],
                pt["n_streamlines"],
            ])
            rf.flush()
            all_times_s.extend(pt["_time_list_s"])
            all_lengths_m.extend(pt["_length_list_m"])
            logger.info(f"finish run {i + 1:>3}/{n_total}  "
                        f"phi={state[0]:>3} deg  theta={state[1]:>3} deg  "
                        f"face={state[2]}   k={k:.4e}  [{status}]  "
                        f"n_sl={pt['n_streamlines']}")
 
    start2 = datetime.datetime.now()
 
    # ---------------------------------------------------------------
    # PHASE 3 - split, fit conductivity tensor and plot
    # ---------------------------------------------------------------
    # Drop failed runs (NaN) and non-physical k values (negative or non-finite).
    # A negative k_eff usually means the andfn solver hit MAX_ITERATIONS without
    # converging and returned a meaningless total flow -- those points would
    # poison the least-squares fit.
    valid = np.isfinite(all_k) & (all_k > 0)
    n_failed = int(np.sum(~valid))
    if n_failed:
        logger.warning(f"{n_failed}/{n_total} simulations failed or returned "
                       f"non-physical k; using {valid.sum()} valid measurements")
    all_directions = all_directions[valid]
    all_k = all_k[valid]
    n_total = len(all_k)
 
    fit_idx, val_idx = split_indices(n_total, val_fraction=0.2)
    fit_directions, val_directions = all_directions[fit_idx], all_directions[val_idx]
    fit_k, val_k = all_k[fit_idx], all_k[val_idx]
    logger.info(f"Fit directions:        {len(fit_directions)}")
    logger.info(f"Validation directions: {len(val_directions)}")
 
    fit_points = fit_directions * fit_k[:, np.newaxis]
    val_points = val_directions * val_k[:, np.newaxis]
    all_points = np.vstack([fit_points, val_points])
 
    A_fit = build_design_matrix(fit_directions)
    result, _, _, _ = np.linalg.lstsq(A_fit, fit_k, rcond=None)
    Kxx, Kxy, Kxz, Kyy, Kyz, Kzz = result
    K_tensor = np.array([
        [Kxx, Kxy, Kxz],
        [Kxy, Kyy, Kyz],
        [Kxz, Kyz, Kzz],
    ])
 
    fit_k_pred = A_fit @ result
    rmse_fit = np.sqrt(np.mean((fit_k - fit_k_pred) ** 2))
    A_val = build_design_matrix(val_directions)
    val_k_pred = A_val @ result
    rmse_val = np.sqrt(np.mean((val_k - val_k_pred) ** 2))
    cond = np.linalg.cond(A_fit)
 
    eigvals, eigvecs = np.linalg.eigh(K_tensor)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
 
    summary_lines = [
        "Run configuration:",
        f"  Simulation date: {start0:%Y-%m-%d}",
        f"  Simulation start time: {start0:%Y-%m-%d %H:%M:%S}",
        f"  Fracture CSV file: {path}",
        f"  Region box: center={BOX_CENTER}, edge length={BOX_SIZE} m "
        f"(volume = {BOX_SIZE**3:.3e} m^3)",
        f"  Rotation grid step: {rotations[1] - rotations[0]} deg",
        f"  Applied heads: low={HEAD_LOW} m, high={HEAD_HIGH} m "
        f"(dH = {HEAD_HIGH - HEAD_LOW} m)",
        "",
        f"Sampling: {len(fit_directions)} fit + {len(val_directions)} validation",
        f"Design matrix condition number: {cond:.2f}  (< 100 is good)",
        "",
        "Fitted 3D conductivity tensor:",
        f"  K = [[{Kxx:.4e}, {Kxy:.4e}, {Kxz:.4e}],",
        f"       [{Kxy:.4e}, {Kyy:.4e}, {Kyz:.4e}],",
        f"       [{Kxz:.4e}, {Kyz:.4e}, {Kzz:.4e}]]",
        "",
        f"  Principal values: k1={eigvals[0]:.4e}, k2={eigvals[1]:.4e}, k3={eigvals[2]:.4e}",
        "  K tensor in principal axes (diagonalized):",
        f"    [[{eigvals[0]:.4e}, {0.0:.4e}, {0.0:.4e}],",
        f"     [{0.0:.4e}, {eigvals[1]:.4e}, {0.0:.4e}],",
        f"     [{0.0:.4e}, {0.0:.4e}, {eigvals[2]:.4e}]]",
    ]
    for i in range(3):
        summary_lines.append(
            f"    v{i+1} = [{eigvecs[0,i]:.4f}, {eigvecs[1,i]:.4f}, {eigvecs[2,i]:.4f}]"
        )
    summary_lines += [
        "",
        f"  RMSE on fit directions:        {rmse_fit:.4e}",
        f"  RMSE on held-out (validation): {rmse_val:.4e}  <- key quality indicator",
    ]
    # ---------------------------------------------------------------
    # SPD-constrained fit via cvxpy: solve the same least-squares but
    # require K ⪰ 0 (positive semidefinite). When the unconstrained fit
    # has a marginally-negative eigenvalue (within RMSE of zero) this
    # typically yields a near-identical answer with the negative eigenvalue
    # pushed to ~0.
    # ---------------------------------------------------------------
    # Rescale so the optimisation variable is O(1): conic solvers use absolute
    # tolerances (~1e-8) that swamp a k ~ 1e-9 objective otherwise, producing
    # a "feasible but meaningless" optimum. Solve in scaled units then undo.
    K_SCALE = 1.0 / max(np.max(np.abs(fit_k)), 1e-300)
    K_var = cp.Variable((3, 3), PSD=True)
    k_vec = cp.hstack([K_var[0, 0], K_var[0, 1], K_var[0, 2],
                       K_var[1, 1], K_var[1, 2], K_var[2, 2]])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A_fit @ k_vec - fit_k * K_SCALE)))
    prob.solve(solver=cp.CLARABEL)
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
        "SPD-constrained fit (K ⪰ 0 via cvxpy):",
        f"  K_spd = [[{K_spd[0,0]:.4e}, {K_spd[0,1]:.4e}, {K_spd[0,2]:.4e}],",
        f"           [{K_spd[0,1]:.4e}, {K_spd[1,1]:.4e}, {K_spd[1,2]:.4e}],",
        f"           [{K_spd[0,2]:.4e}, {K_spd[1,2]:.4e}, {K_spd[2,2]:.4e}]]",
        f"  Principal values (SPD): k1={eigvals_spd[0]:.4e}, "
        f"k2={eigvals_spd[1]:.4e}, k3={eigvals_spd[2]:.4e}",
    ]
    for i in range(3):
        summary_lines.append(
            f"    v{i+1}_spd = [{eigvecs_spd[0,i]:.4f}, "
            f"{eigvecs_spd[1,i]:.4f}, {eigvecs_spd[2,i]:.4f}]"
        )
    summary_lines += [
        f"  RMSE (fit, SPD):        {rmse_fit_spd:.4e}",
        f"  RMSE (validation, SPD): {rmse_val_spd:.4e}",
        f"  cvxpy status: {prob.status}",
    ]
 
    # ---------------------------------------------------------------
    # Axis-aligned sanity check: compare the fit's diagonal with the
    # direct DFN measurements along the x, y, z axes (phi=0, theta=0).
    # The axis-aligned simulations should match Kxx, Kyy, Kzz closely;
    # large disagreement means the tensor fit is underconstrained along
    # that axis (e.g. near-2D DFN with almost no vertical flow).
    # ---------------------------------------------------------------
    def _lookup_axis_k(directions, k_values, axis_vec, tol=1e-6):
        """Return the measured k for the direction matching axis_vec (any sign)."""
        dots = np.abs(directions @ axis_vec)
        idx = int(np.argmax(dots))
        return k_values[idx] if dots[idx] > 1 - tol else np.nan
 
    k_axis = {
        "x": _lookup_axis_k(all_directions, all_k, np.array([1.0, 0, 0])),
        "y": _lookup_axis_k(all_directions, all_k, np.array([0, 1.0, 0])),
        "z": _lookup_axis_k(all_directions, all_k, np.array([0, 0, 1.0])),
    }
    summary_lines += [
        "",
        "Axis-aligned sanity check (direct DFN vs fitted diagonal):",
        f"  k_x (direct) = {k_axis['x']:.4e}   Kxx (lsq) = {Kxx:.4e}   "
        f"Kxx (SPD) = {K_spd[0,0]:.4e}",
        f"  k_y (direct) = {k_axis['y']:.4e}   Kyy (lsq) = {Kyy:.4e}   "
        f"Kyy (SPD) = {K_spd[1,1]:.4e}",
        f"  k_z (direct) = {k_axis['z']:.4e}   Kzz (lsq) = {Kzz:.4e}   "
        f"Kzz (SPD) = {K_spd[2,2]:.4e}",
        "  (Large |direct - fit| along an axis indicates the tensor is",
        "   underconstrained in that direction -- e.g. near-2D fracture set.)",
    ]
 
    # ---------------------------------------------------------------
    # Paper-style hydraulic conductivity ellipsoid (Wang et al. 2023,
    # eqs. 7-9). In principal-axis coordinates the directional conductivity
    # obeys 1/k(n) = cos^2(theta_i)/K_i, so the locus of points
    #   p = sqrt(k(n)) * n
    # lies on the ellipsoid x^2/K1 + y^2/K2 + z^2/K3 = 1 (semi-axes sqrt(K_i)).
    # The paper's normalized relative error Er compares, for each measured
    # direction, the measured k against the fitted ellipsoid's predicted k:
    #   Er = (1/N) * sum |1 - k_meas(n) / k_fit(n)|
    # Er < 0.3 is the paper's KREV acceptance threshold.
    # ---------------------------------------------------------------
    def _paper_k_pred(directions, K_mat):
        """k(n) = n . K . n along each direction (n must be unit)."""
        return np.einsum("ij,jk,ik->i", directions, K_mat, directions)
 
    def _paper_er(k_meas, k_pred):
        good = np.isfinite(k_pred) & (k_pred > 0) & (k_meas > 0)
        if not np.any(good):
            return np.nan
        return float(np.mean(np.abs(1.0 - k_meas[good] / k_pred[good])))
 
    k_pred_lsq_all = _paper_k_pred(all_directions, K_tensor)
    k_pred_spd_all = _paper_k_pred(all_directions, K_spd)
    er_lsq = _paper_er(all_k, k_pred_lsq_all)
    er_spd = _paper_er(all_k, k_pred_spd_all)
    er_lsq_val = _paper_er(val_k, _paper_k_pred(val_directions, K_tensor))
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
 
    summary_text = "\n".join(summary_lines)
    for line in summary_lines:
        logger.info(line)
 
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    np.savez(os.path.join(run_dir, "tensor.npz"),
             K=K_tensor, eigvals=eigvals, eigvecs=eigvecs,
             rmse_fit=rmse_fit, rmse_val=rmse_val, cond=cond,
             K_spd=K_spd, eigvals_spd=eigvals_spd, eigvecs_spd=eigvecs_spd,
             rmse_fit_spd=rmse_fit_spd, rmse_val_spd=rmse_val_spd,
             k_axis_x=k_axis["x"], k_axis_y=k_axis["y"], k_axis_z=k_axis["z"],
             er_lsq=er_lsq, er_spd=er_spd,
             er_lsq_val=er_lsq_val, er_spd_val=er_spd_val,
             fit_directions=fit_directions, fit_k=fit_k,
             val_directions=val_directions, val_k=val_k,
             fit_idx=fit_idx, val_idx=val_idx)
 
    # --- 3D scatter plot (saved from several viewpoints; a single static PNG
    # of a 3D scatter is unreadable).
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(fit_points[:, 0], fit_points[:, 1], fit_points[:, 2],
               c="red", marker="o", s=10, label="Fit measurements")
    ax.scatter(val_points[:, 0], val_points[:, 1], val_points[:, 2],
               c="orange", marker="^", s=40, label="Validation measurements")
    ax.set_xlabel("Conductivity x")
    ax.set_ylabel("Conductivity y")
    ax.set_zlabel("Conductivity z")
    ax.set_title("3D Effective Conductivity - Rotation Grid Sampling")
    ax.legend()
    for azim in (30, 120, 210, 300):
        ax.view_init(elev=20, azim=azim)
        fig.savefig(os.path.join(run_dir, f"scatter_3d_azim{azim:03d}.png"),
                    dpi=200, bbox_inches="tight")
    ax.view_init(elev=20, azim=30)
    plt.show(block=False)
 
    # --- Interactive 3D scatter (PyVista) -- same data as the matplotlib
    # version, but exported as HTML + VTK so you can orbit/zoom in a browser.
    fit_cloud = pv.PolyData(fit_points)
    fit_cloud["k_eff"] = fit_k
    val_cloud = pv.PolyData(val_points)
    val_cloud["k_eff"] = val_k
    fit_cloud.save(os.path.join(run_dir, "scatter_fit.vtk"))
    val_cloud.save(os.path.join(run_dir, "scatter_val.vtk"))
 
    plotter_scatter = pv.Plotter(title="3D Effective Conductivity Scatter")
    plotter_scatter.add_points(fit_cloud, color="red", point_size=8,
                               render_points_as_spheres=True,
                               label="Fit measurements")
    plotter_scatter.add_points(val_cloud, color="orange", point_size=12,
                               render_points_as_spheres=True,
                               label="Validation measurements")
    plotter_scatter.add_legend()
    plotter_scatter.add_axes()
    plotter_scatter.show_grid(xtitle="Conductivity x",
                              ytitle="Conductivity y",
                              ztitle="Conductivity z")
    plotter_scatter.camera_position = "iso"
    try:
        plotter_scatter.export_html(os.path.join(run_dir, "scatter_3d.html"))
    except Exception as exc:
        logger.warning(f"scatter_3d.html export failed: "
                       f"{type(exc).__name__}: {exc}")
    plotter_scatter.show(screenshot=os.path.join(run_dir, "scatter_3d.png"))
 
    # --- Convex hull surface
    hull = ConvexHull(all_points)
    faces = hull.simplices
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    surface = pv.PolyData(all_points, pv_faces)
    # Save VTK before any plotter window opens -- a PolyData.save() can't be
    # lost if the render window later crashes or is closed early.
    surface.save(os.path.join(run_dir, "convex_hull.vtk"))
    plotter_hull = pv.Plotter(title="Convex Hull")
    plotter_hull.add_mesh(surface, color="cyan", opacity=0.5, show_edges=True,
                          label="Convex Hull Surface")
    plotter_hull.add_points(fit_points, color="red", point_size=10,
                            label="Fit measurements")
    plotter_hull.add_points(val_points, color="orange", point_size=12,
                            label="Validation measurements")
    plotter_hull.add_legend()
    plotter_hull.add_axes()
    plotter_hull.camera_position = "iso"
    try:
        plotter_hull.export_html(os.path.join(run_dir, "convex_hull.html"))
    except Exception as exc:
        logger.warning(f"convex_hull.html export failed: "
                       f"{type(exc).__name__}: {exc}")
    plotter_hull.show(screenshot=os.path.join(run_dir, "convex_hull.png"))
 
    # --- Fitted "k-surface": r(n) = (n . K . n) * n, so the radius along any
    # direction equals the predicted conductivity in that direction. This is
    # the surface the least-squares fit is actually trying to match against
    # the measured `k * n` points.
    sphere = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
    sphere_pts = np.array(sphere.points)
    k_dir = np.einsum("ij,ij->i", sphere_pts @ K_tensor, sphere_pts)
    ellipsoid = sphere.copy()
    ellipsoid.points = sphere_pts * k_dir[:, None]
    ellipsoid.save(os.path.join(run_dir, "ellipsoid.vtk"))
 
    plotter = pv.Plotter(title="Fitted k-surface (ellipsoid)")
    plotter.add_mesh(ellipsoid, color="lightblue", opacity=0.4,
                     label="Fitted k-surface")
    plotter.add_points(fit_points, color="red", point_size=8,
                       label="Fit measurements")
    plotter.add_points(val_points, color="orange", point_size=12,
                       label="Validation measurements")
    labels = ["k1 (max)", "k2 (mid)", "k3 (min)"]
    for i, color in enumerate(["red", "green", "blue"]):
        axis = eigvecs[:, i] * eigvals[i]
        plotter.add_arrows(np.zeros((1, 3)), axis.reshape(1, 3),
                           color=color, mag=1)
        plotter.add_point_labels(
            axis.reshape(1, 3),
            [f"{labels[i]} = {eigvals[i]:.4e}"],
            font_size=12, text_color=color, bold=True,
        )
    plotter.add_axes()
    plotter.camera_position = "iso"
    try:
        plotter.export_html(os.path.join(run_dir, "ellipsoid.html"))
    except Exception as exc:
        logger.warning(f"ellipsoid.html export failed: "
                       f"{type(exc).__name__}: {exc}")
    plotter.show(screenshot=os.path.join(run_dir, "ellipsoid.png"))
 
    # --- Paper-style hydraulic conductivity ellipsoid (Wang et al. 2023).
    # Radius along n is sqrt(k(n)); measurement points are sqrt(k)*n. Semi-axes
    # are sqrt(K_i) along the principal directions. Use the SPD fit so the
    # ellipsoid is guaranteed real (negative K_i -> imaginary semi-axis).
    eigvals_plot = np.clip(eigvals_spd, 0.0, None)
    paper_pts = (sphere_pts * np.sqrt(eigvals_plot)) @ eigvecs_spd.T
    paper_ellipsoid = sphere.copy()
    paper_ellipsoid.points = paper_pts
    paper_ellipsoid.save(os.path.join(run_dir, "paper_ellipsoid.vtk"))
 
    sqrt_fit_points = fit_directions * np.sqrt(fit_k)[:, None]
    sqrt_val_points = val_directions * np.sqrt(val_k)[:, None]
 
    plotter_paper = pv.Plotter(title="Paper-style conductivity ellipsoid (sqrt(k))")
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
        f"(Er < 0.3 -> KREV acceptance, Wang et al. 2023)",
        font_size=10, position="upper_left",
    )
    plotter_paper.add_legend()
    plotter_paper.add_axes()
    plotter_paper.camera_position = "iso"
    try:
        plotter_paper.export_html(os.path.join(run_dir, "paper_ellipsoid.html"))
    except Exception as exc:
        logger.warning(f"paper_ellipsoid.html export failed: "
                       f"{type(exc).__name__}: {exc}")
    plotter_paper.show(screenshot=os.path.join(run_dir, "paper_ellipsoid.png"))
 
    # --- Triangulated shell surface of the measured ellipsoid
    # Triangulate on the unit directions so every measurement point is used,
    # then apply those face indices back to the original 3-D coordinates.
    if len(all_points) >= 4:
        shell_norms = np.linalg.norm(all_points, axis=1, keepdims=True)
        shell_unit  = all_points / np.where(shell_norms > 0, shell_norms, 1.0)
        shell_hull  = ConvexHull(shell_unit)
        shell_faces = np.hstack([
            np.full((len(shell_hull.simplices), 1), 3, dtype=np.intp),
            shell_hull.simplices,
        ]).ravel()
        shell_mesh = pv.PolyData(all_points, shell_faces)
        shell_mesh.save(os.path.join(run_dir, "shell.vtk"))
 
        point_cloud = pv.PolyData(all_points)
        point_cloud["point_id"] = np.arange(1, len(all_points) + 1, dtype=int)
 
        plotter_shell = pv.Plotter(title="Measured Ellipsoid Shell")
        plotter_shell.add_mesh(shell_mesh, color="lightsteelblue",
                               opacity=0.30, show_edges=True)
        plotter_shell.add_points(point_cloud, color="darkred", point_size=10,
                                 label="Measurement points")
        plotter_shell.add_scalar_bar(title="Point ID")
        plotter_shell.add_axes()
        plotter_shell.add_title(
            f"Measured Ellipsoid - Triangulated Shell "
            f"({len(all_points)} points)")
        plotter_shell.camera_position = "iso"
        try:
            plotter_shell.export_html(os.path.join(run_dir, "shell.html"))
        except Exception as exc:
            logger.warning(f"shell.html export failed: "
                           f"{type(exc).__name__}: {exc}")
        plotter_shell.show(screenshot=os.path.join(run_dir, "shell.png"))

    # ---------------------------------------------------------------
    # PHASE 4 - BTC and trace-length CDFs from all rotation states
    # ---------------------------------------------------------------
    logger.info("---- BREAKTHROUGH CURVES (all rotation states) ----")
    _yr = 3600.0 * 24.0 * 365.0
    if len(all_times_s) > 0:
        time_yr    = np.array(all_times_s) / _yr
        length_m   = np.array(all_lengths_m)
        an_len     = length_m[length_m > BOX_SIZE]

        sorted_t = np.sort(time_yr)
        cdf_t    = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
        fig_btc, ax_btc = plt.subplots(figsize=(8, 6))
        ax_btc.plot(sorted_t, cdf_t, color="red", label="AnDFN (all rotations)")
        ax_btc.set_xscale("log")
        ax_btc.set_xlabel("Travel Time [years]", fontsize=16,
                          fontname="Times New Roman")
        ax_btc.set_ylabel("Cumulative Distribution Function", fontsize=16,
                          fontname="Times New Roman")
        ax_btc.set_title(
            f"Breakthrough Curve — {len(all_rotation_states)} rotation states, "
            f"{len(all_times_s)} streamlines",
            fontsize=12,
        )
        ax_btc.legend(prop={"size": 14, "family": "Times New Roman"})
        ax_btc.grid(True)
        ax_btc.tick_params(colors="black", labelsize=14,
                           labelfontfamily="Times New Roman")
        fig_btc.tight_layout()
        fig_btc.savefig(
            os.path.join(run_dir, "breakthrough_curve.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig_btc)
        logger.info("Saved breakthrough_curve.png")

        if len(an_len) > 0:
            sorted_l = np.sort(an_len)
            cdf_l    = np.arange(1, len(sorted_l) + 1) / len(sorted_l)
            fig_tl, ax_tl = plt.subplots(figsize=(8, 6))
            ax_tl.plot(sorted_l, cdf_l, color="red", label="AnDFN (all rotations)")
            ax_tl.set_xlabel("Trace length [m]", fontsize=16,
                              fontname="Times New Roman")
            ax_tl.set_ylabel("Cumulative Distribution Function", fontsize=16,
                              fontname="Times New Roman")
            ax_tl.set_title(
                f"Trace Length CDF — paths longer than {BOX_SIZE:.0f} m",
                fontsize=12,
            )
            ax_tl.legend(prop={"size": 14, "family": "Times New Roman"})
            ax_tl.grid(True)
            ax_tl.tick_params(colors="black", labelsize=14,
                               labelfontfamily="Times New Roman")
            fig_tl.tight_layout()
            fig_tl.savefig(
                os.path.join(run_dir, "trace_length_cdf.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig_tl)
            logger.info("Saved trace_length_cdf.png")
        else:
            logger.info("No traces longer than box edge — trace_length_cdf.png skipped")
    else:
        logger.info("No streamlines collected across any rotation state — BTC skipped")

    end = datetime.datetime.now()
    logger.info(f"Program ended at {end}")
    logger.info(f"Time elapsed: {end - start0}")
    logger.info(f"\t- loading + phase 1: {start1 - start0}")
    logger.info(f"\t- DFN simulations:   {start2 - start1}")
    logger.info(f"\t- fit + plotting:    {end - start2}")
    logger.info(f"All artifacts written to: {run_dir}")
    logger.info("All done! Close the plot windows to exit.")
 
    # Block on the matplotlib window so the 3D scatter stays visible. The
    # PyVista windows are already shown one-by-one above (each call to
    # `.show()` blocks until you close that window).
    plt.show()
 
 