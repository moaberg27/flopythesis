"""
ECM particle tracking with anisotropic hydraulic conductivity tensor.
 
"""
 
import datetime
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import flopy
 
 
def fix_fortran_scientific(path):
    """Fix missing 'E' in Fortran scientific notation (e.g. 1.23-252 → 1.23E-252)."""
    text = Path(path).read_text()
    fixed = re.sub(r'(\d)([-+]\d{2,3})(?=\s|$)', r'\1E\2', text, flags=re.MULTILINE)
    if fixed != text:
        Path(path).write_text(fixed)
 
# -----------------------------------------------------------------------------
# 1. INPUT PARAMETERS
# -----------------------------------------------------------------------------
 
# Principal hydraulic conductivities [m/s]
K1 = 1.23e-08
K2 = 1.03e-08
K3 = 2.87e-09
 
# Principal directions (eigenvectors in the GLOBAL frame of the DFN).
# Used directly: no 45 deg rotation is applied.
v1 = np.array([0.2021, -0.9766, -0.073])
v2 = np.array([-0.9793,  -0.2025, -0.002])
v3 = np.array([ 0.0127, -0.072,  0.9973])
 
# Porosity
porosity = 5.10e-4
 
# Domain and grid
Lx = Ly = Lz = 500.0    # domain side length [m]
dx = dy = dz = 25.0     # cell size [m]
nlay = int(Lz / dz)    
nrow = int(Ly / dy)
ncol = int(Lx / dx)
 
# Boundary conditions
h_low  = 100.0
h_high = 200.0
 
# Particle tracking
sim_time_years = 1.0e6
seconds_per_year = 365.25 * 24 * 3600
 
# Output workspace: ECM_runs/PT_runs/<timestamp>
run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(r"C:\Users\SEAM94860\FLOPY\finalflopy") / "ECM_runs" / "PT_runs" / run_ts
out_dir.mkdir(parents=True, exist_ok=True)
 
# Executables
flopy_root = Path(__file__).resolve().parent
mf6_exe = str((flopy_root / "mf6.exe").resolve())
mp7_exe = str((flopy_root / "mpath7.exe").resolve())
 
if not Path(mf6_exe).exists():
    raise FileNotFoundError(f"MODFLOW 6 executable not found: {mf6_exe}")
if not Path(mp7_exe).exists():
    raise FileNotFoundError(f"MODPATH 7 executable not found: {mp7_exe}")
 
 
# -----------------------------------------------------------------------------
# 2. ROTATION ANGLES FROM EIGENVECTORS
# -----------------------------------------------------------------------------
 
def euler_angles_from_eigenvectors(v1, v2, v3):
    """Return MODFLOW 6 (angle1, angle2, angle3) in degrees from the three
    principal direction vectors. Convention: V = Rz(a1) . Ry(a2) . Rx(a3)."""
    V = np.column_stack([v1, v2, v3])
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
 
    cos_a2 = np.sqrt(V[0, 0] ** 2 + V[1, 0] ** 2)
    a2 = np.arctan2(-V[2, 0], cos_a2)
    if cos_a2 > 1e-6:
        a1 = np.arctan2(V[1, 0], V[0, 0])
        a3 = np.arctan2(V[2, 1], V[2, 2])
    else:
        a1 = 0.0
        a3 = np.arctan2(-V[1, 2], V[1, 1])
 
    return np.degrees([a1, a2, a3])
 
 
angle1, angle2, angle3 = euler_angles_from_eigenvectors(v1, v2, v3)
 
print("Principal conductivities:")
print(f"  K1 = {K1:.3e}   K2 = {K2:.3e}   K3 = {K3:.3e}  m/s")
print(f"Rotation angles (deg): {angle1:.3f}, {angle2:.3f}, {angle3:.3f}")
 
 
# -----------------------------------------------------------------------------
# 3. HELPERS
# -----------------------------------------------------------------------------
 
def make_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf
 
 
def plot_cdf(data, label, **kwargs):
    sd, cdf = make_cdf(data)
    plt.plot(sd, cdf, label=label, **kwargs)
 
 
def run_simulation(axis):
    """Run one MODFLOW 6 + MODPATH 7 simulation with flow along `axis`
    ('x', 'y', or 'z'). Returns travel times (years)."""
    print(f"\n========== SIMULATION: flow along {axis} ==========")
 
    sim_dir = out_dir / f"sim_{axis}"
    sim_dir.mkdir()
    ws = str(sim_dir)
 
    name = f"ecm_{axis}"
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=ws, exe_name=mf6_exe, version="mf6"
    )
    flopy.mf6.ModflowTdis(
        sim, time_units="SECONDS", nper=1, perioddata=[(1.0, 1, 1.0)]
    )
    flopy.mf6.ModflowIms(
        sim,
        complexity="MODERATE",
        outer_dvclose=1e-6,
        inner_dvclose=1e-7,
        linear_acceleration="BICGSTAB",
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
 
    top = Lz
    botm = [Lz - dz * (k + 1) for k in range(nlay)]
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay, nrow=nrow, ncol=ncol,
        delr=dx, delc=dy,
        top=top, botm=botm,
        length_units="METERS",
    )
 
    flopy.mf6.ModflowGwfic(gwf, strt=h_high)
 
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_specific_discharge=True,
        xt3doptions=True,
        icelltype=0,
        k=K1, k22=K2, k33=K3,
        angle1=angle1, angle2=angle2, angle3=angle3,
    )
 
    # Build CHD cells for the inlet/outlet faces of the chosen axis.
    chd_data = []
    if axis == "x":
        for k in range(nlay):
            for i in range(nrow):
                chd_data.append([(k, i, 0),         h_high])
                chd_data.append([(k, i, ncol - 1),  h_low])
    elif axis == "y":
        for k in range(nlay):
            for j in range(ncol):
                chd_data.append([(k, 0, j),         h_high])
                chd_data.append([(k, nrow - 1, j),  h_low])
    elif axis == "z":
        for i in range(nrow):
            for j in range(ncol):
                chd_data.append([(0, i, j),         h_high])
                chd_data.append([(nlay - 1, i, j),  h_low])
    else:
        raise ValueError(axis)
 
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_data, save_flows=True)
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
 
    print("Running MODFLOW 6 ...")
    sim.write_simulation()
    success, _ = sim.run_simulation(silent=True)
    if not success:
        raise RuntimeError("MODFLOW 6 did not converge.")
 
    # Total flow rate from CHD budget
    cbc = flopy.utils.CellBudgetFile(os.path.join(ws, f"{name}.cbc"))
    chd_rec = cbc.get_data(text="CHD", kstpkper=(0, 0))[0]
    q_chd = np.asarray(chd_rec["q"], dtype=float)
    Q_in = q_chd[q_chd > 0].sum()
    print(f"Total flow Q = {Q_in:.4e} m^3/s")
 
    # ---- Particle tracking: one particle per inlet cell ----
    particle_locs = []
    if axis == "x":
        for k in range(nlay):
            for i in range(nrow):
                particle_locs.append([(k, i, 0), 0.05, 0.5, 0.5])
    elif axis == "y":
        for k in range(nlay):
            for j in range(ncol):
                particle_locs.append([(k, 0, j), 0.5, 0.05, 0.5])
    elif axis == "z":
        for i in range(nrow):
            for j in range(ncol):
                particle_locs.append([(0, i, j), 0.5, 0.5, 0.95])
 
    pdata = flopy.modpath.ParticleData(
        partlocs=[p[0] for p in particle_locs],
        structured=True,
        localx=[p[1] for p in particle_locs],
        localy=[p[2] for p in particle_locs],
        localz=[p[3] for p in particle_locs],
        drape=0,
    )
    pg = flopy.modpath.ParticleGroup(
        particlegroupname="inflow",
        particledata=pdata,
        filename=f"{name}.sloc",
    )
 
    mp_name = f"{name}_mp"
    mp = flopy.modpath.Modpath7(
        modelname=mp_name, flowmodel=gwf, exe_name=mp7_exe, model_ws=ws,
    )
    flopy.modpath.Modpath7Bas(mp, porosity=porosity)
    flopy.modpath.Modpath7Sim(
        mp,
        simulationtype="endpoint",
        trackingdirection="forward",
        weaksinkoption="pass_through",
        weaksourceoption="pass_through",
        referencetime=0.0,
        stoptimeoption="specified",
        stoptime=sim_time_years * seconds_per_year,
        particlegroups=[pg],
    )
 
    print("Running MODPATH 7 ...")
    mp.write_input()
    success, _ = mp.run_model(silent=True)
    if not success:
        raise RuntimeError("MODPATH 7 did not run.")
 
    # Read endpoints (fix Fortran missing-E scientific notation first)
    mpend_path = os.path.join(ws, f"{mp_name}.mpend")
    fix_fortran_scientific(mpend_path)
    endpoints = flopy.utils.EndpointFile(mpend_path).get_alldata()
 
    node_term = np.asarray(endpoints["node"], dtype=int)
    k_t, i_t, j_t = np.unravel_index(node_term, (nlay, nrow, ncol))
    if axis == "x":
        exited = j_t == (ncol - 1)
    elif axis == "y":
        exited = i_t == (nrow - 1)
    else:  # z
        exited = k_t == (nlay - 1)
 
    travel_time_s = endpoints["time"][exited]
    travel_time_yr = travel_time_s / seconds_per_year
 
    print(f"Particles released: {len(endpoints)}    "
          f"reached outlet: {int(exited.sum())}")
 
    # Save particle data and per-axis BTC plot
    pd.DataFrame({"travel_time_yr": travel_time_yr}).to_csv(
        sim_dir / f"particles_{axis}.csv", index=False
    )
 
    plt.figure(figsize=(8, 6))
    if len(travel_time_yr) > 0:
        plot_cdf(travel_time_yr, label=f"ECM ({axis})", color="blue")
    plt.xscale("log")
    plt.xlabel("Travel time [years]", fontsize=14)
    plt.ylabel("Cumulative fraction", fontsize=14)
    plt.title(f"BTC - flow along {axis}   Q = {Q_in:.2e} m^3/s")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(sim_dir / f"btc_{axis}.png", dpi=150)
    plt.close()
 
    return {"axis": axis, "time_yr": travel_time_yr, "Q": Q_in}
 
 
# -----------------------------------------------------------------------------
# 4. RUN ALL THREE SIMULATIONS
# -----------------------------------------------------------------------------
 
results = {}
for axis in ["x", "y", "z"]:
    results[axis] = run_simulation(axis)
 
 
# -----------------------------------------------------------------------------
# 5. FINAL COMPARISON PLOT
# -----------------------------------------------------------------------------
 
print("\n---- FINAL COMPARISON PLOT ----")
colors = {"x": "red", "y": "green", "z": "blue"}
 
plt.figure(figsize=(8, 6))
for axis, r in results.items():
    if len(r["time_yr"]) > 0:
        plot_cdf(r["time_yr"], label=f"flow {axis}", color=colors[axis])
plt.xscale("log")
plt.xlabel("Travel time [years]", fontsize=14)
plt.ylabel("Cumulative fraction", fontsize=14)
plt.title("ECM BTC comparison - flow along x, y, z")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "btc_comparison.png", dpi=150)
plt.show()
 
print(f"\nAll outputs saved in: {out_dir}")
print("Done.")
 
 