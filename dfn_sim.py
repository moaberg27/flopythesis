"""
Notes
-----
This is an example of a model.
"""

import datetime
import os

import numpy as np
import pandas as pd

import andfn
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import RBFInterpolator

# Add rotation matrix and precompute directions functions
def rotation_matrix(axis, angle_deg):
    """Return the 3x3 rotation matrix for rotating `angle_deg` around `axis`."""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0,  0], [0, c, -s], [0, s,  c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")

FACE_NORMALS_LOCAL = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
], dtype=float)

def precompute_directions(rotations):
    step = rotations[2] - rotations[0]
    phi_angles = list(range(0, 360, step))
    theta_angles = list(range(0, 180, step))

    seen = set()
    unique_directions = []
    rotation_states = []

    for phi_deg in phi_angles:
        for theta_deg in theta_angles:
            R = rotation_matrix('x', theta_deg) @ rotation_matrix('z', phi_deg)
            for face_idx, n_local in enumerate(FACE_NORMALS_LOCAL):
                n_lab = R @ n_local
                key = tuple(np.round(n_lab, 8))
                if key not in seen:
                    seen.add(key)
                    unique_directions.append(n_lab)
                    rotation_states.append((phi_deg, theta_deg, face_idx))

    return np.array(unique_directions), rotation_states

# Define principal conductivity values
K_PRINCIPAL = np.array([3.0, 3.0, 3.0])  # kx, ky, kz

# Integrate rotation and ellipsoid fitting logic
rotations = [0, 15, 30, 45, 60, 75]
all_directions, all_rotation_states = precompute_directions(rotations)

def split_directions(directions, rotation_states, val_fraction=0.2, seed=42):
    """
    Split precomputed directions into fit / validation sets.

    Parameters
    ----------
    directions : (N, 3) ndarray
    rotation_states : list of tuples
    val_fraction : float
    seed : int

    Returns
    -------
    fit_dirs, val_dirs, fit_states, val_states
    """
    n_total = len(directions)
    n_val = max(1, int(n_total * val_fraction))

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    fit_idx = indices[:-n_val]
    val_idx = indices[-n_val:]

    fit_states = [rotation_states[i] for i in fit_idx]
    val_states = [rotation_states[i] for i in val_idx]

    return directions[fit_idx], directions[val_idx], fit_states, val_states

fit_directions, val_directions, fit_states, val_states = split_directions(
    all_directions, all_rotation_states, val_fraction=0.2,
)

def run_dfn_simulation(phi_deg, theta_deg, face_index):
    """
    Run a single DFN flow simulation and return the effective conductivity.

    Parameters
    ----------
    phi_deg : float
        Azimuth rotation of the domain around Z (degrees).
    theta_deg : float
        Elevation rotation of the domain around X (degrees).
    face_index : int
        Which face pair to impose the pressure gradient on:
          0 → gradient along local X (faces ±X)
          1 → gradient along local Y (faces ±Y)
          2 → gradient along local Z (faces ±Z)

    Returns
    -------
    k_eff : float
        Effective (equivalent) conductivity measured for this configuration.

    NOTE: This is a synthetic placeholder using k_eff = n^T K n + noise.
    Replace this entire function body with your real DFN solver call, e.g.:
        result = your_dfn_solver.run(domain, rotation=(phi_deg, theta_deg),
                                     gradient_axis=face_index)
        return result.equivalent_conductivity
    """
    R = rotation_matrix('x', theta_deg) @ rotation_matrix('z', phi_deg)
    n_lab = R @ FACE_NORMALS_LOCAL[face_index]
    k_eff = (K_PRINCIPAL[0] * n_lab[0]**2 +
             K_PRINCIPAL[1] * n_lab[1]**2 +
             K_PRINCIPAL[2] * n_lab[2]**2)
    noise = 1.0 - np.random.rand() / 5   # ±20% variability
    return k_eff * noise

fit_k = np.array([run_dfn_simulation(*state) for state in fit_states])
val_k = np.array([run_dfn_simulation(*state) for state in val_states])

fit_points = fit_directions * fit_k[:, np.newaxis]
val_points = val_directions * val_k[:, np.newaxis]
all_points = np.vstack([fit_points, val_points])

def build_design_matrix(directions):
    """
    Build the least-squares design matrix for the symmetric tensor fit.

    Each row encodes:  k = Kxx*nx^2 + 2Kxy*nx*ny + 2Kxz*nx*nz + Kyy*ny^2 + 2Kyz*ny*nz + Kzz*nz^2
    so A @ [Kxx, Kxy, Kxz, Kyy, Kyz, Kzz] = k_meas
    """
    nx, ny, nz = directions[:, 0], directions[:, 1], directions[:, 2]
    return np.column_stack([
        nx**2,
        2 * nx * ny,
        2 * nx * nz,
        ny**2,
        2 * ny * nz,
        nz**2,
    ])

A_fit = build_design_matrix(fit_directions)
result, _, _, _ = np.linalg.lstsq(A_fit, fit_k, rcond=None)
Kxx, Kxy, Kxz, Kyy, Kyz, Kzz = result

K_tensor = np.array([
    [Kxx, Kxy, Kxz],
    [Kxy, Kyy, Kyz],
    [Kxz, Kyz, Kzz],
])

# Plot ellipsoid
sphere = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
sphere_pts = np.array(sphere.points)
eigvals, eigvecs = np.linalg.eigh(K_tensor)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

ellipsoid_pts = (eigvecs @ np.diag(eigvals) @ sphere_pts.T).T
ellipsoid = sphere.copy()
ellipsoid.points = ellipsoid_pts

plotter = pv.Plotter()
plotter.add_mesh(ellipsoid, color='lightblue', opacity=0.4, label='Fitted Ellipsoid')
plotter.add_points(fit_points, color='red', point_size=8, label='Fit measurements')
plotter.add_points(val_points, color='orange', point_size=12, label='Validation measurements')
plotter.add_axes()
plotter.show()

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

    ncoef = 10 * 0 + 10
    nint = ncoef * 2

    start0 = datetime.datetime.now()
    print("\n---- IMPORT DFN ----")
    print(f"Program started at {start0}")

    # load the geometry
    dfn_org = andfn.DFN("DFN test FracMan", discharge_int=50)

    # name ="p32_case11"
    path = os.path.join(r"C:\Users\SEAM94860\FLOPY\finalflopy\flopythesis", "fracs_connected_properties.csv")
    #path = os.path.join(r"C:\Users\SEAM94860\FLOPY\finalflopy\flopythesis", "15000fracs.csv")
    reload = False

    # Check if it exist saved
    print("DFN  importing from file")
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
        remove_isolated=False
    )


    # TODO: add the rotation loop
    for r in range(1):
        dfn = andfn.DFN("Copy", discharge_int=50)
        dfn.add_fracture(dfn_org.fractures)

        print("Adding constant head boundary conditions")
        # Add Region Box boundary
        head0 = 100
        head1 = 200
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
        regbox.rotate(angle=75, axis=[1, 0, 0])
        regbox.rotate(angle=15, axis=[0, 0, 1])
        reg_fracs_in, reg_fracs_out = regbox.check_fractures(dfn.fractures, tree=dfn.tree)

        print(f"Number of fractures in the DFN: {len(dfn.fractures)}")
        dfn.delete_fracture(reg_fracs_out)
        print(f"Number of fractures after deleting those outside the box: {len(dfn.fractures)}")

        regbox.frac_intersections(dfn.fractures, face="front", head=head0)
        regbox.frac_intersections(dfn.fractures, face="back", head=head1)

        dfn.check_connectivity()        

        dfn.set_kwargs(COEF_RATIO=0.001, MAX_ITERATIONS=30, MAX_NCOEF=200, MAX_ERROR=5e-4)

        start1 = datetime.datetime.now()
        print("\n---- SOLVE THE DFN ----")
        dfn.solve(unconsolidate=True)

        start2 = datetime.datetime.now()

        print("\n---- GET FLOWS ----")
        total_flow = [
            np.abs(e.q) for e in regbox.elements if isinstance(e, ConstantHeadLine)
        ]
        sum_flows = np.sum(total_flow) / 2
        print(f"Total flow through the box: {sum_flows:.2e} m^3/s")

        print("\n---- PLOTTING ----")
        p1 = dfn.initiate_plotter(title=True, off_screen=False, scale=1, axis=True)

        dfn.plot_fractures_head(
            p1, 40, 10, opacity=1, contour=True
        )  # , limits=[200, 400], debug=False)
        regbox.plot(p1)



        p1.show()

    end = datetime.datetime.now()
    print(f"\n\nProgram ended at {end}")
    print(f"Time elapsed: {end - start0}")
    print(f"\t-generating: \t{start1 - start0}")
    print(f"\t-solving: \t\t\t{start2 - start1}")
    print(f"\t-plotting: \t\t{end - start2}")

    print("All done!")