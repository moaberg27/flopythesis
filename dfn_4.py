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
BOX_SIZE   = 250.0       # cubic domain edge length (m)
HEAD_LOW   = 100.0
HEAD_HIGH  = 200.0
 
 
def _snapshot_elements(fractures):
    """Capture the element list of every fracture so we can restore it later.
 
    andfn shares Fracture objects between `dfn_org` and per-run DFN copies.
    `delete_fracture` and `frac_intersections` mutate those objects (strip or
    append elements). Copying the list references (not the elements) is enough
    to reverse those mutations between runs.
    """
    return [list(f.elements) for f in fractures]
 
 
def _restore_elements(fractures, snapshot):
    for f, original in zip(fractures, snapshot):
        f.elements = list(original)
 
 
def run_dfn_simulation(dfn_org, snapshot, phi_deg, theta_deg, face_index):
    """Run one DFN flow simulation and return the effective conductivity.
 
    The region box is rotated by phi around Z and then theta around X
    (matching R = Rx(theta) @ Rz(phi) applied to local face normals).
    Constant-head BCs are imposed on the face pair selected by
    `face_index`: 0 -> left/right (local X), 1 -> front/back (local Y),
    2 -> bottom/top (local Z).
 
    Returns NaN if the solver fails (e.g. singular discharge matrix for a
    degenerate/disconnected orientation) so the main loop can continue.
    Always restores the per-fracture element lists on the way out so
    successive runs start from the pristine imported state.
    """
    try:
        dfn = andfn.DFN("Copy", discharge_int=50)
        dfn.add_fracture(list(dfn_org.fractures))
 
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
                       MAX_NCOEF=200, MAX_ERROR=5e-4)
        dfn.solve(unconsolidate=True)
 
        Q  = regbox.get_total_flow() / 2.0
        dH = HEAD_HIGH - HEAD_LOW
        L  = BOX_SIZE
        A  = BOX_SIZE * BOX_SIZE
        return Q * L / (A * dH) # effective conductivity from Darcy's law: Q = k A dH / L
    except Exception as exc:
        logger.warning(
            f"simulation failed "
            f"(phi={phi_deg}, theta={theta_deg}, face={face_index}): "
            f"{type(exc).__name__}: {exc}"
        )
        return np.nan
    finally:
        # Undo the mutations `delete_fracture` / `frac_intersections` made to
        # the shared Fracture objects, so the next run starts pristine.
        _restore_elements(dfn_org.fractures, snapshot)
 
 
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, "fracs_connected_properties.csv"),
        os.path.join(os.path.dirname(script_dir), "fracs_connected_properties.csv"),
    ]
    path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
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
 
    # Snapshot the original element lists once; restored after every run so
    # the shared Fracture objects aren't progressively corrupted.
    original_elements = _snapshot_elements(dfn_org.fractures)
    original_fractures = list(dfn_org.fractures)
 
    n_total = len(all_rotation_states)
    all_k = np.empty(n_total)
    results_path = os.path.join(run_dir, "simulation_results.csv")
    with open(results_path, "w", newline="") as rf:
        rw = csv.writer(rf)
        rw.writerow(["i", "phi_deg", "theta_deg", "face",
                     "nx", "ny", "nz", "k_eff", "status"])
        for i, state in enumerate(all_rotation_states):
            k = run_dfn_simulation(dfn_org, original_elements, *state)
            # `delete_fracture` only touches the per-run DFN's list, but keep the
            # org DFN's fracture list pristine as well in case anything else
            # iterates over it.
            dfn_org.fractures = list(original_fractures)
            all_k[i] = k
            nx, ny, nz = all_directions[i]
            status = "ok" if (np.isfinite(k) and k > 0) else "invalid"
            rw.writerow([i, state[0], state[1], state[2],
                         f"{nx:.8f}", f"{ny:.8f}", f"{nz:.8f}",
                         f"{k:.6e}", status])
            rf.flush()
            logger.info(f"  run {i + 1:>3}/{n_total}  "
                        f"phi={state[0]:>3} deg  theta={state[1]:>3} deg  "
                        f"face={state[2]}   k={k:.4e}  [{status}]")
 
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
        f"  {np.diag(eigvals)}",
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
    # require K >> 0 (positive semidefinite). When the unconstrained fit
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
        "SPD-constrained fit (K >> 0 via cvxpy):",
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
    plt.close(fig)
 
    # --- Convex hull surface
    hull = ConvexHull(all_points)
    faces = hull.simplices
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    surface = pv.PolyData(all_points, pv_faces)
    plotter_hull = pv.Plotter(off_screen=True)
    plotter_hull.add_mesh(surface, color="cyan", opacity=0.5, show_edges=True,
                          label="Convex Hull Surface")
    plotter_hull.add_points(fit_points, color="red", point_size=10,
                            label="Fit measurements")
    plotter_hull.add_points(val_points, color="orange", point_size=12,
                            label="Validation measurements")
    plotter_hull.add_legend()
    plotter_hull.add_axes()
    plotter_hull.camera_position = "iso"
    plotter_hull.show(screenshot=os.path.join(run_dir, "convex_hull.png"),
                      auto_close=False)
    plotter_hull.export_html(os.path.join(run_dir, "convex_hull.html"))
    surface.save(os.path.join(run_dir, "convex_hull.vtk"))
    try:
        plotter_hull.close()
    except Exception:
        pass
 
    # --- Fitted "k-surface": r(n) = (n . K . n) * n, so the radius along any
    # direction equals the predicted conductivity in that direction. This is
    # the surface the least-squares fit is actually trying to match against
    # the measured `k * n` points.
    sphere = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
    sphere_pts = np.array(sphere.points)
    k_dir = np.einsum("ij,ij->i", sphere_pts @ K_tensor, sphere_pts)
    ellipsoid = sphere.copy()
    ellipsoid.points = sphere_pts * k_dir[:, None]
 
    plotter = pv.Plotter(off_screen=True)
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
    plotter.show(screenshot=os.path.join(run_dir, "ellipsoid.png"),
                 auto_close=False)
    plotter.export_html(os.path.join(run_dir, "ellipsoid.html"))
    ellipsoid.save(os.path.join(run_dir, "ellipsoid.vtk"))
    try:
        plotter.close()
    except Exception:
        pass
 
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
 
        point_cloud = pv.PolyData(all_points)
        point_cloud["point_id"] = np.arange(1, len(all_points) + 1, dtype=int)
 
        plotter_shell = pv.Plotter(off_screen=True)
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
        plotter_shell.show(screenshot=os.path.join(run_dir, "shell.png"),
                           auto_close=False)
        plotter_shell.export_html(os.path.join(run_dir, "shell.html"))
        shell_mesh.save(os.path.join(run_dir, "shell.vtk"))
        try:
            plotter_shell.close()
        except Exception:
            pass
 
    end = datetime.datetime.now()
    logger.info(f"Program ended at {end}")
    logger.info(f"Time elapsed: {end - start0}")
    logger.info(f"\t- loading + phase 1: {start1 - start0}")
    logger.info(f"\t- DFN simulations:   {start2 - start1}")
    logger.info(f"\t- fit + plotting:    {end - start2}")
    logger.info(f"All artifacts written to: {run_dir}")
    logger.info("All done!")
 
 
 