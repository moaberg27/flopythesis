"""
plot_shell_and_ellipsoid.py
---------------------------
Read a simulation_results.csv produced by the DFN flow solver and generate:
 
    (1) the triangulated measurement surface  (shell of sqrt(k)*n cloud)
    (2) the fitted Wang-style hydraulic conductivity ellipsoid
    (3) both overlaid in the same view
 
The CSV must have columns: nx, ny, nz, k_eff, status.
Rows with status != "ok" or non-positive k_eff are discarded.
 
Outputs (in `out_dir`):
    measurement_shell.vtk    triangulated shell mesh
    fitted_ellipsoid.vtk     fitted ellipsoid mesh
    shell.png                screenshot of plot (1)
    ellipsoid.png            screenshot of plot (2)
    shell_and_ellipsoid.png  screenshot of plot (3)
 
Usage:
    python plot_shell_and_ellipsoid.py simulation_results.csv [out_dir]
 
Requires:  numpy, pandas, scipy, pyvista
Optional:  cvxpy  (used for the PSD-constrained fit; falls back to plain
                   LSQ if not installed or if the unconstrained fit is
                   already PSD).
"""
 
import os
import sys
 
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import ConvexHull
 
try:
    import cvxpy as cp
    HAVE_CVXPY = True
except ImportError:
    HAVE_CVXPY = False
 
 
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_measurements(csv_path):
    """Load (n, k) pairs from a simulation_results.csv, dropping invalid rows."""
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    k = df["k_eff"].to_numpy(dtype=float)
    df = df[np.isfinite(k) & (k > 0)]
    n = df[["nx", "ny", "nz"]].to_numpy(dtype=float)
    n /= np.linalg.norm(n, axis=1, keepdims=True)  # safety re-normalisation
    k = df["k_eff"].to_numpy(dtype=float)
    return n, k
 
 
# ---------------------------------------------------------------------------
# Tensor fitting
# ---------------------------------------------------------------------------
def build_design_matrix(directions):
    """A @ [Kxx, Kxy, Kxz, Kyy, Kyz, Kzz]  =  k_meas    (Eq. 12, lab frame)."""
    nx, ny, nz = directions[:, 0], directions[:, 1], directions[:, 2]
    return np.column_stack([
        nx ** 2, 2 * nx * ny, 2 * nx * nz,
        ny ** 2, 2 * ny * nz, nz ** 2,
    ])
 
 
def _x_to_K(x):
    """Re-assemble symmetric 3x3 K from the 6-component fit vector."""
    return np.array([[x[0], x[1], x[2]],
                     [x[1], x[3], x[4]],
                     [x[2], x[4], x[5]]])
 
 
def fit_K_lsq(directions, k):
    """Plain unconstrained least-squares fit of K via np.linalg.lstsq."""
    A = build_design_matrix(directions)
    x, *_ = np.linalg.lstsq(A, k, rcond=None)
    return _x_to_K(x)
 
 
def fit_K_spd(directions, k):
    """PSD-constrained fit of K via cvxpy/CLARABEL.
 
    The variable and right-hand side are rescaled by K_SCALE so that the
    optimiser sees an O(1) problem; conic solvers use absolute tolerances
    of order 1e-8 and would otherwise treat k ~ 1e-9 as numerical noise.
    """
    if not HAVE_CVXPY:
        raise ImportError("cvxpy is required for the SPD fit.")
    A = build_design_matrix(directions)
    K_SCALE = 1.0 / max(np.max(np.abs(k)), 1e-300)
    K_var = cp.Variable((3, 3), PSD=True)
    k_vec = cp.hstack([K_var[0, 0], K_var[0, 1], K_var[0, 2],
                       K_var[1, 1], K_var[1, 2], K_var[2, 2]])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ k_vec - k * K_SCALE)))
    prob.solve(solver=cp.CLARABEL)
    K = np.array(K_var.value) / K_SCALE
    return 0.5 * (K + K.T)
 
 
def sorted_eigh(K):
    """Eigendecomposition of K with eigenvalues sorted descending (K1 >= K2 >= K3)."""
    eigvals, eigvecs = np.linalg.eigh(K)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]
 
 
# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------
def triangulated_shell(directions, k):
    """Triangulate the sqrt(k)*n cloud using the unit-direction sphere connectivity.
 
    The convex hull of the unit vectors n_i gives a clean triangulation of the
    sphere; the same simplex indices are then applied to the (anisotropic)
    sqrt(k)*n positions to produce a closed surface mesh whose vertices are
    exactly the measurement points.
    """
    sqrt_pts = directions * np.sqrt(k)[:, None]
    hull = ConvexHull(directions)
    faces = np.column_stack([
        np.full(len(hull.simplices), 3, dtype=np.intp),
        hull.simplices,
    ]).ravel()
    return pv.PolyData(sqrt_pts, faces), sqrt_pts
 
 
def fitted_ellipsoid(K, sphere_resolution=50):
    """Build the Wang ellipsoid (semi-axes sqrt(Ki)) from the fitted K.
 
    Returns a PyVista mesh plus the (sorted) eigenvalues and eigenvectors of K.
    Eigenvalues are clipped to non-negative before sqrt, so a tiny negative
    drift from an unconstrained LSQ does not produce a NaN ellipsoid.
    """
    eigvals, eigvecs = sorted_eigh(K)
    eigvals_plot = np.clip(eigvals, 0.0, None)
    sphere = pv.Sphere(radius=1.0,
                       theta_resolution=sphere_resolution,
                       phi_resolution=sphere_resolution)
    sphere_pts = np.array(sphere.points)
    pts = (sphere_pts * np.sqrt(eigvals_plot)) @ eigvecs.T
    mesh = sphere.copy()
    mesh.points = pts
    return mesh, eigvals, eigvecs
 
 
# ---------------------------------------------------------------------------
# Quality metric (Wang Eq. 9)
# ---------------------------------------------------------------------------
def normalised_relative_error(directions, k_meas, K):
    """Er = mean( |1 - k_meas/k_fit| ),  k_fit(n) = n^T K n   (Wang Eq. 9)."""
    k_pred = np.einsum("ij,jk,ik->i", directions, K, directions)
    good = np.isfinite(k_pred) & (k_pred > 0) & (k_meas > 0)
    if not np.any(good):
        return float("nan")
    return float(np.mean(np.abs(1.0 - k_meas[good] / k_pred[good])))
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- paste your paths here ----
    csv_path = r"C:\Users\SEAM94860\FLOPY\finalflopy\simulation_results_dfn6.csv"
    # --------------------------------

    if not csv_path:
        if len(sys.argv) < 2:
            sys.exit(__doc__)
        csv_path = sys.argv[1]

    dfn_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(r"C:\Users\SEAM94860\FLOPY\finalflopy\plots", dfn_name)
    os.makedirs(out_dir, exist_ok=True)
    # --- 1. Load + fit
    n, k = load_measurements(csv_path)
    print(f"Loaded {len(k)} valid measurements from {csv_path}")
    print(f"  k range: {k.min():.3e}  to  {k.max():.3e}  m/s")
 
    K_lsq = fit_K_lsq(n, k)
    eigvals_lsq, _ = sorted_eigh(K_lsq)
 
    # Use the SPD fit if cvxpy is available AND the LSQ fit returned a
    # negative eigenvalue (otherwise plain LSQ is already PSD and faster).
    if eigvals_lsq[-1] < 0 and HAVE_CVXPY:
        print("  unconstrained LSQ has a negative eigenvalue;"
              " switching to PSD-constrained fit (cvxpy/CLARABEL)")
        K = fit_K_spd(n, k)
        fit_label = "SPD-constrained"
    else:
        K = K_lsq
        fit_label = "least-squares"
 
    eigvals, eigvecs = sorted_eigh(K)
    er = normalised_relative_error(n, k, K)
 
    print(f"\nFit method: {fit_label}")
    print("Fitted K (m/s):")
    print(np.array2string(K, formatter={"float": lambda v: f"{v: .4e}"}))
    print(f"\nPrincipal hydraulic conductivities:")
    print(f"  K1 = {eigvals[0]: .3e}   (max)")
    print(f"  K2 = {eigvals[1]: .3e}")
    print(f"  K3 = {eigvals[2]: .3e}   (min)")
    accept = "<  0.3, KREV accepted" if er < 0.3 else ">= 0.3, KREV rejected"
    print(f"\nNormalised relative error  Er = {er:.4f}  ({accept})")
 
    # --- 2. Build meshes
    shell_mesh, sqrt_pts = triangulated_shell(n, k)
    ellipsoid_mesh, _, _ = fitted_ellipsoid(K)
 
    shell_mesh.save(os.path.join(out_dir, "measurement_shell.vtk"))
    ellipsoid_mesh.save(os.path.join(out_dir, "fitted_ellipsoid.vtk"))
 
    # Common helpers for the three plots
    eigvals_plot = np.clip(eigvals, 0.0, None)
    axis_colors = ["red", "green", "blue"]
 
    def add_principal_axes(plotter, with_labels=True):
        names = ["K1 (max)", "K2 (mid)", "K3 (min)"]
        for i, color in enumerate(axis_colors):
            semi = eigvecs[:, i] * np.sqrt(eigvals_plot[i])
            plotter.add_arrows(np.zeros((1, 3)), semi.reshape(1, 3),
                               color=color, mag=1,
                               label=f"{names[i]} = {eigvals_plot[i]:.3e}")
            if with_labels:
                plotter.add_point_labels(
                    semi.reshape(1, 3),
                    [f"sqrt(K{i+1}) = {np.sqrt(eigvals_plot[i]):.3e}"],
                    font_size=11, text_color=color, bold=True,
                )
 
    # --- 3. Plot (1): triangulated measurement surface alone
    p1 = pv.Plotter(title="Triangulated measurement surface  (sqrt(k_meas) * n)")
    p1.add_mesh(shell_mesh, color="lightcoral", opacity=0.55,
                show_edges=True, edge_color="darkred",
                label="Triangulated sqrt(k_meas)*n surface")
    p1.add_points(sqrt_pts, color="darkred", point_size=8,
                  render_points_as_spheres=True,
                  label="Measurement points sqrt(k)*n")
    p1.add_legend()
    p1.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
    p1.camera_position = "iso"
    p1.show(screenshot=os.path.join(out_dir, "shell.png"), auto_close=False)
    try:
        p1.export_html(os.path.join(out_dir, "shell.html"))
    except Exception as exc:
        print(f"  shell.html export failed: {exc}")
    p1.close()
 
    # --- Plot (2): fitted ellipsoid alone
    p2 = pv.Plotter(title="Fitted hydraulic conductivity ellipsoid  (semi-axes sqrt(Ki))")
    p2.add_mesh(ellipsoid_mesh, color="lightgreen", opacity=0.55,
                label=f"Fitted ellipsoid ({fit_label})")
    add_principal_axes(p2, with_labels=True)
    p2.add_text(f"Er = {er:.3f}",
                font_size=10, position="upper_left")
    p2.add_legend()
    p2.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
    p2.camera_position = "iso"
    p2.show(screenshot=os.path.join(out_dir, "ellipsoid.png"), auto_close=False)
    try:
        p2.export_html(os.path.join(out_dir, "ellipsoid.html"))
    except Exception as exc:
        print(f"  ellipsoid.html export failed: {exc}")
    p2.close()
 
    # --- Plot (3): both overlaid
    p3 = pv.Plotter(title="Measurement shell vs fitted ellipsoid")
    p3.add_mesh(shell_mesh, color="lightcoral", opacity=0.30,
                show_edges=True, edge_color="darkred",
                label="Triangulated  sqrt(k_meas) * n  surface")
    p3.add_mesh(ellipsoid_mesh, color="lightgreen", opacity=0.30,
                label=f"Fitted ellipsoid  ({fit_label})")
    p3.add_points(sqrt_pts, color="darkred", point_size=7,
                  render_points_as_spheres=True,
                  label="Measurement points")
    add_principal_axes(p3, with_labels=False)
    p3.add_text(f"Er = {er:.3f}   ({'< 0.3 KREV OK' if er < 0.3 else '>= 0.3 KREV NO'})",
                font_size=10, position="upper_left")
    p3.add_legend()
    p3.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
    p3.camera_position = "iso"
    p3.show(screenshot=os.path.join(out_dir, "shell_and_ellipsoid.png"), auto_close=False)
    try:
        p3.export_html(os.path.join(out_dir, "shell_and_ellipsoid.html"))
    except Exception as exc:
        print(f"  shell_and_ellipsoid.html export failed: {exc}")
    p3.close()
 
    # --- Plot (4): Wang et al. 2023 style — grey shell + grey ellipsoid + red dots
    #     Mirrors the plotter_combo block in andfn.py.
    p4 = pv.Plotter(title="Measurement shell vs fitted ellipsoid (Wang et al. 2023 style)")
    p4.add_mesh(shell_mesh, color="lightsteelblue", opacity=0.30,
                show_edges=True, edge_color="steelblue",
                label="Triangulated sqrt(k_meas)*n surface")
    p4.add_mesh(ellipsoid_mesh, color="lightgreen", opacity=0.30,
                label=f"Fitted ellipsoid ({fit_label})")
    p4.add_points(sqrt_pts, color="darkred", point_size=8,
                  render_points_as_spheres=True,
                  label="Measurement points sqrt(k)*n")
    add_principal_axes(p4, with_labels=False)
    p4.add_text(
        f"Er = {er:.3f}   ({'< 0.3 KREV OK' if er < 0.3 else '>= 0.3 KREV NO'})",
        font_size=10, position="upper_left",
    )
    p4.add_legend()
    p4.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
    p4.camera_position = "iso"
    p4.show(screenshot=os.path.join(out_dir, "shell_and_ellipsoid_wang.png"), auto_close=False)
    try:
        p4.export_html(os.path.join(out_dir, "shell_and_ellipsoid_wang.html"))
    except Exception as exc:
        print(f"  shell_and_ellipsoid_wang.html export failed: {exc}")
    p4.close()
 
    print(f"\nArtifacts written to: {out_dir}")
 
 
if __name__ == "__main__":
    main()
 
 