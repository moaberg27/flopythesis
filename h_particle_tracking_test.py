# Set up the Environment
import os
import sys
import gc
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

proj_root = Path.cwd().parent.parent
script_dir = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load the fitted 3D conductivity tensor from the rotation analysis
# ---------------------------------------------------------------------------
TENSOR_CSV = script_dir / "csv_files" / "tensor_sim_one.csv"


def tensor_to_npf_params(csv_path):
    """Read fitted tensor CSV and return MODFLOW 6 NPF parameters.

    MODFLOW 6 angle convention (XT3D):
      angle1 = rotation about z   (azimuth of k1 in the xy-plane)
      angle2 = rotation about x'  (dip of k1 below the xy-plane)
      angle3 = rotation about y'' (roll / bank)
    """
    df = pd.read_csv(csv_path)
    row = df.iloc[0]

    k1, k2, k3 = float(row["k1"]), float(row["k2"]), float(row["k3"])
    v1 = np.array([float(row["v1_x"]), float(row["v1_y"]), float(row["v1_z"])])
    v2 = np.array([float(row["v2_x"]), float(row["v2_y"]), float(row["v2_z"])])

    # angle1: azimuth of k1 projected onto the xy-plane
    angle1 = np.degrees(np.arctan2(v1[1], v1[0]))

    # angle2: dip of k1 below horizontal
    angle2 = np.degrees(np.arcsin(np.clip(-v1[2], -1.0, 1.0)))

    # angle3: rotation of k2/k3 around k1 after applying angle1 & angle2
    cos1, sin1 = np.cos(np.radians(angle1)), np.sin(np.radians(angle1))
    cos2, sin2 = np.cos(np.radians(angle2)), np.sin(np.radians(angle2))
    y_prime = np.array([-sin1, cos1, 0.0])
    z_prime = np.array([-cos1 * sin2, -sin1 * sin2, cos2])
    angle3 = np.degrees(np.arctan2(np.dot(v2, z_prime), np.dot(v2, y_prime)))

    return {"k": k1, "k22": k2, "k33": k3,
            "angle1": angle1, "angle2": angle2, "angle3": angle3}

# Manually specified NPF parameters (if tensor CSV is not available)
npf_params = {"k": 2.0, "k22": 2.0, "k33": 2.0,
                  "angle1": 0.0, "angle2": 0.0, "angle3": 0.0}

# if TENSOR_CSV.exists():
#     npf_params = tensor_to_npf_params(TENSOR_CSV)
# else:
#     print(f"WARNING: {TENSOR_CSV} not found — falling back to isotropic K=2.0")
#     npf_params = {"k": 2.0, "k22": 2.0, "k33": 2.0,
#                   "angle1": 0.0, "angle2": 0.0, "angle3": 0.0}

# Use executables from the script directory
flopy_root = Path(__file__).resolve().parent
mf_exe = (flopy_root / "mf6.exe").resolve()
gridgen_exe = (flopy_root / "gridgen.exe").resolve()
mp_exe = (flopy_root / "mp7.exe").resolve()

if not mf_exe.exists():
    raise FileNotFoundError(f"MODFLOW executable not found: {mf_exe}")
if not gridgen_exe.exists():
    raise FileNotFoundError(f"Gridgen executable not found: {gridgen_exe}")
if not mp_exe.exists():
    raise FileNotFoundError(f"MODPATH executable not found: {mp_exe}")

# run installed version of flopy or add local path
try:
    import flopy
except:
    sys.path.append(proj_root)
    import flopy

# temporary directory
temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
workspace = Path(temp_dir.name)

# Creation of the model grid
Lx = 5000.0
Ly = 5000.0
nlay = 1
nrow = 100
ncol = 100
delr = Lx / ncol
delc = Ly / nrow
top = 50
botm = [0]
ms = flopy.modflow.Modflow(exe_name=str(mf_exe))
dis5 = flopy.modflow.ModflowDis(
    ms,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=delr,
    delc=delc,
    top=top,
    botm=botm,
)

# Create the Gridgen object
from flopy.utils.gridgen import Gridgen

model_name = "mp7_3D"
model_ws = workspace / "mp7_3D" / "mf6"
gridgen_ws = model_ws / "gridgen"
g = Gridgen(ms.modelgrid, model_ws=gridgen_ws, exe_name=str(gridgen_exe))

# Refine the grid – all zones centred at (Lx/2, Ly/2)
cx_model, cy_model = Lx / 2, Ly / 2

rf0shp = gridgen_ws / "rf0"
xmin, xmax = cx_model - 500, cx_model + 500
ymin, ymax = cy_model - 500, cy_model + 500
rfpoly = [
    [
        list(
            reversed(
                [
                    (xmin, ymin),
                    (xmax, ymin),
                    (xmax, ymax),
                    (xmin, ymax),
                    (xmin, ymin),
                ]
            )
        )
    ]
]
g.add_refinement_features(rfpoly, "polygon", 1, range(nlay))

rf1shp = gridgen_ws / "rf1"
xmin, xmax = cx_model - 250, cx_model + 250
ymin, ymax = cy_model - 250, cy_model + 250
rfpoly = [
    [
        list(
            reversed(
                [
                    (xmin, ymin),
                    (xmax, ymin),
                    (xmax, ymax),
                    (xmin, ymax),
                    (xmin, ymin),
                ]
            )
        )
    ]
]
g.add_refinement_features(rfpoly, "polygon", 2, range(nlay))

rf2shp = gridgen_ws / "rf2"
xmin, xmax = cx_model - 100, cx_model + 100
ymin, ymax = cy_model - 100, cy_model + 100
rfpoly = [
    [
        list(
            reversed(
                [
                    (xmin, ymin),
                    (xmax, ymin),
                    (xmax, ymax),
                    (xmin, ymax),
                    (xmin, ymin),
                ]
            )
        )
    ]
]
g.add_refinement_features(rfpoly, "polygon", 3, range(nlay))

# Show model grid 
fig = plt.figure(figsize=(5, 5), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1)
mm = flopy.plot.PlotMapView(model=ms)
mm.plot_grid()
flopy.plot.plot_shapefile(rf0shp, ax=ax, facecolor="yellow", edgecolor="none")
flopy.plot.plot_shapefile(rf1shp, ax=ax, facecolor="pink", edgecolor="none")
flopy.plot.plot_shapefile(rf2shp, ax=ax, facecolor="red", edgecolor="none")

# Build the refined grid.
g.build(verbose=False)

# Show the refined grid
fig = plt.figure(figsize=(5, 5), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
g.plot(ax, linewidth=0.5)

# Show all pending figures.
plt.show()

# Extract the refined grid’s properties.
gridprops = g.get_gridprops_disv()
ncpl = gridprops["ncpl"]
top = gridprops["top"]
botm = gridprops["botm"]
nvert = gridprops["nvert"]
vertices = gridprops["vertices"]
cell2d = gridprops["cell2d"]

# Create simulation
sim = flopy.mf6.MFSimulation(
    sim_name=model_name, version="mf6", exe_name=str(mf_exe), sim_ws=model_ws
)

# create tdis package
tdis_rc = [(1.0, 1, 1.0)] # (length of time step, number of steps, time step multiplier)
tdis = flopy.mf6.ModflowTdis(
    sim, pname="tdis", time_units="DAYS", perioddata=tdis_rc
)

# create gwf model
gwf = flopy.mf6.ModflowGwf(
    sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
)
gwf.name_file.save_flows = True

# create iterative model solution and register the gwf model with it
ims = flopy.mf6.ModflowIms(
    sim,
    pname="ims",
    print_option="SUMMARY",
    complexity="SIMPLE",
    outer_dvclose=1.0e-5,
    outer_maximum=100,
    under_relaxation="NONE",
    inner_maximum=100,
    inner_dvclose=1.0e-6,
    rcloserecord=0.1,
    linear_acceleration="BICGSTAB",
    scaling_method="NONE",
    reordering_method="NONE",
    relaxation_factor=0.99,
)
sim.register_ims_package(ims, [gwf.name])

# disv
disv = flopy.mf6.ModflowGwfdisv(
    gwf,
    nlay=nlay,
    ncpl=ncpl,
    top=top,
    botm=botm,
    nvert=nvert,
    vertices=vertices,
    cell2d=cell2d,
)

# initial conditions
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=100.0)

# node property flow — full 3D anisotropy via XT3D
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    xt3doptions=[("xt3d")],
    icelltype=[0],  # confined: prevents cells going dry at the well
    k=[npf_params["k"]],
    k22=[npf_params["k22"]],
    k33=[npf_params["k33"]],
    angle1=[npf_params["angle1"]],
    angle2=[npf_params["angle2"]],
    angle3=[npf_params["angle3"]],
    save_specific_discharge=True,
)

# well – placed at the model centre, inside the most-refined zone
wellpoints = [(Lx / 2, Ly / 2)]
welcells = g.intersect(wellpoints, "point", 0)

# welspd = flopy.mf6.ModflowGwfwel.stress_period_data.empty(gwf, maxbound=1, aux_vars=['iface'])
welspd = [[(0, icpl), -5000, 0] for icpl in welcells["nodenumber"]]
wel = flopy.mf6.ModflowGwfwel(
    gwf, print_input=True, auxiliary=[("iface",)], stress_period_data=welspd
)

# constant head boundary conditions on all 4 sides
h1 = 100.0  # left   (x = 0)
h2 = 100.0  # right  (x = Lx)
h3 = 100.0  # bottom (y = 0)
h4 = 100.0  # top    (y = Ly)

# Identify boundary cells (outermost ring of cells) for each layer.
# On a quadtree DISV grid cell centres are offset inward by half the cell
# width, so we use delr/2 and delc/2 as tolerances to capture the first
# column/row of cells regardless of refinement level.
# cell2d format: [icell2d, xc, yc, ncvert, iv1, iv2, ...]
xc = np.array([c[1] for c in cell2d])  # cell-centre x
yc = np.array([c[2] for c in cell2d])  # cell-centre y

left_cells   = np.where(xc <= delr / 2)[0]
right_cells  = np.where(xc >= Lx - delr / 2)[0]
bottom_cells = np.where(yc <= delc / 2)[0]
top_cells    = np.where(yc >= Ly - delc / 2)[0]

chdspd = []
seen = set()
for lay in range(nlay):
    for icpl in left_cells:
        if (lay, icpl) not in seen:
            seen.add((lay, icpl))
            chdspd.append([(lay, icpl), h1])
    for icpl in right_cells:
        if (lay, icpl) not in seen:
            seen.add((lay, icpl))
            chdspd.append([(lay, icpl), h2])
    for icpl in bottom_cells:
        if (lay, icpl) not in seen:
            seen.add((lay, icpl))
            chdspd.append([(lay, icpl), h3])
    for icpl in top_cells:
        if (lay, icpl) not in seen:
            seen.add((lay, icpl))
            chdspd.append([(lay, icpl), h4])

chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd)

chd_cells = np.unique(np.concatenate([left_cells, right_cells, bottom_cells, top_cells]))

# output control
oc = flopy.mf6.ModflowGwfoc(
    gwf,
    pname="oc",
    budget_filerecord="{}.cbb".format(model_name),
    head_filerecord="{}.hds".format(model_name),
    headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
)

sim.write_simulation()

success, buff = sim.run_simulation(silent=True, report=True)
for line in buff:
    print(line)
assert success, "mf6 failed to run"

# Import and plot the results
fname = os.path.join(model_ws, model_name + ".disv.grb")
grd = flopy.mf6.utils.MfGrdFile(fname, verbose=False)
mg = grd.modelgrid
ibd = np.zeros((ncpl), dtype=int)
ibd[welcells["nodenumber"]] = 1
ibd[chd_cells] = 2
ibd = np.ma.masked_equal(ibd, 0)
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
pmv = flopy.plot.PlotMapView(modelgrid=mg, ax=ax)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
cmap = mpl.colors.ListedColormap(
    [
        "r",
        "g",
    ]
)
pc = pmv.plot_array(ibd, cmap=cmap, edgecolor="gray")
t = ax.set_title("Boundary Conditions\n")

# Show all pending figures (no file export).
plt.show()

# Import and plot the results
fname = os.path.join(model_ws, model_name + ".hds")
hdobj = flopy.utils.HeadFile(fname)
head = hdobj.get_data()
head.shape

# Mask HDRY values (1e+30) that MODFLOW writes for dry/inactive cells
head_plot = np.ma.masked_where(head > 1e+20, head)

# Plot the head distribution in layer 3
ilay = 0
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mm = flopy.plot.PlotMapView(modelgrid=mg, ax=ax, layer=ilay)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
pc = mm.plot_array(head_plot[:, 0, :], cmap="jet", edgecolor="black")
hmin = float(head_plot[ilay, 0, :].min())
hmax = float(head_plot[ilay, 0, :].max())
levels = np.linspace(hmin, hmax, 20)
cs = mm.contour_array(head_plot[:, 0, :], colors="white", levels=levels)
plt.clabel(cs, fmt="%.1f", colors="white", fontsize=11)
cb = plt.colorbar(pc, shrink=0.5)
t = ax.set_title(
    "Model Layer {}; hmin={:6.2f}, hmax={:6.2f}".format(ilay + 1, hmin, hmax)
)

plt.show()

# 2D vertical cross section through the pumping well.
wel_icpl = int(welcells["nodenumber"][0])
wel_k = int(welspd[0][0][0])
wel_x = float(mg.xcellcenters[wel_icpl])
wel_y = float(mg.ycellcenters[wel_icpl])
wel_z = float(mg.zcellcenters[wel_k, wel_icpl])

xsec_line = {"line": [(0.0, wel_y), (Lx, wel_y)]}

fig = plt.figure(figsize=(10, 5), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1)
xsect = flopy.plot.PlotCrossSection(modelgrid=mg, line=xsec_line, ax=ax)
xsect.plot_grid(color="0.35", lw=0.6)
xsect.plot_bc(package=wel, color="red")

# Add explicit marker/guide for the well location in the section view.
ax.axvline(wel_x, color="red", lw=1.0, ls="--", alpha=0.7)
ax.scatter(
    [wel_x],
    [wel_z],
    c="red",
    s=70,
    edgecolors="black",
    linewidths=0.5,
    zorder=10,
    label="Pumping well",
)
ax.set_title(f"Vertical Cross Section at y={wel_y:.1f} m (through pumping well)")
ax.set_xlabel("x (m)")
ax.set_ylabel("elevation (m)")
ax.legend(loc="lower right")

plt.show()

# Inspect model cells and vertices

# zoom area
xmin, xmax = 2000, 4500
ymin, ymax = 5400, 7500

mg.get_cell_vertices
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mm = flopy.plot.PlotMapView(modelgrid=mg, ax=ax)
v = mm.plot_grid(edgecolor="black")
t = ax.set_title("Model Cells and Vertices (one-based)\n")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

verts = mg.verts
ax.plot(verts[:, 0], verts[:, 1], "bo")
for i in range(ncpl):
    x, y = verts[i, 0], verts[i, 1]
    if xmin <= x <= xmax and ymin <= y <= ymax:
        ax.annotate(str(i + 1), verts[i, :], color="b")

xc, yc = mg.get_xcellcenters_for_layer(0), mg.get_ycellcenters_for_layer(0)
for i in range(ncpl):
    x, y = xc[i], yc[i]
    ax.plot(x, y, "ro")
    if xmin <= x <= xmax and ymin <= y <= ymax:
        ax.annotate(str(i + 1), (x, y), color="r")

plt.show()

#Define names for the MODPATH 7 simulations

mp_namea = model_name + "a_mp"
mp_nameb = model_name + "b_mp"

#Create particles for the pathline and timeseries analysis

# 30-particle ring in neighboring cells around the well cell
# (particles cannot be placed inside the strong-sink well cell on a DISV grid)
well_cell_icpl = int(welcells["nodenumber"][0])
cx = float(mg.xcellcenters[well_cell_icpl])
cy = float(mg.ycellcenters[well_cell_icpl])
xc_cells = np.array(mg.xcellcenters, dtype=float)
yc_cells = np.array(mg.ycellcenters, dtype=float)

nodew = well_cell_icpl  # global node number for the well cell (1 layer)

N_particles = 30
r_ring = 20.0  # physical radius of the release ring [m], inside the refined zone
theta = np.linspace(0, 2 * np.pi, N_particles, endpoint=False)
ring_x = cx + r_ring * np.cos(theta)
ring_y = cy + r_ring * np.sin(theta)

plocs = []
for px, py in zip(ring_x, ring_y):
    d = (xc_cells - px) ** 2 + (yc_cells - py) ** 2
    d[well_cell_icpl] = np.inf  # exclude the well cell itself
    plocs.append(int(np.argmin(d)))

localx = [0.5] * N_particles
localy = [0.5] * N_particles
localz = [0.5] * N_particles

# create particle data
pa = flopy.modpath.ParticleData(
    plocs,
    structured=False,
    localx=localx,
    localy=localy,
    localz=localz,
    drape=0,
)

# create backward particle group
fpth = mp_namea + ".sloc"
pga = flopy.modpath.ParticleGroup(
    particlegroupname="BACKWARD1", particledata=pa, filename=fpth
)

#Create particles for endpoint analysis

facedata = flopy.modpath.FaceDataType(
    drape=0,
    verticaldivisions1=10,
    horizontaldivisions1=10,
    verticaldivisions2=10,
    horizontaldivisions2=10,
    verticaldivisions3=10,
    horizontaldivisions3=10,
    verticaldivisions4=10,
    horizontaldivisions4=10,
    rowdivisions5=0,
    columndivisions5=0,
    rowdivisions6=4,
    columndivisions6=4,
)
pb = flopy.modpath.NodeParticleData(subdivisiondata=facedata, nodes=nodew)

# create forward particle group
fpth = mp_nameb + ".sloc"
pgb = flopy.modpath.ParticleGroupNodeTemplate(
    particlegroupname="BACKWARD2", particledata=pb, filename=fpth
)

# Create and run the pathline and timeseries analysis model.

# create modpath files
mp = flopy.modpath.Modpath7(
    modelname=mp_namea, flowmodel=gwf, exe_name=str(mp_exe), model_ws=model_ws
)
flopy.modpath.Modpath7Bas(mp, porosity=0.3)
flopy.modpath.Modpath7Sim(
    mp,
    simulationtype="combined",
    trackingdirection="backward",
    weaksinkoption="pass_through",
    weaksourceoption="pass_through",
    referencetime=0.0,
    stoptimeoption="extend",
    timepointdata=[500, 1000.0],
    particlegroups=pga,
)

# write modpath datasets
mp.write_input()

# run modpath
success, buff = mp.run_model(silent=True, report=True)
if not success:
    for line in buff:
        print(line)
    joined = "\n".join(buff).lower()
    if "normal termination" not in joined:
        raise RuntimeError("MODPATH 7 failed to run; see output above.")
for line in buff:
    print(line)

# Fix Fortran-style floats missing the 'E' (e.g. "0.123-111" -> "0.123E-111")
import re

def _fix_fortran_floats(fpath):
    # Use latin-1 so every byte round-trips safely (avoids UnicodeDecodeError on Windows)
    raw = Path(fpath).read_text(encoding='latin-1')
    # Insert missing 'E' before Fortran-style exponents (e.g. 0.12345678-103 -> 0.12345678E-103)
    fixed = re.sub(r'(\d)([+-]\d{3})', r'\1E\2', raw)
    Path(fpath).write_text(fixed, encoding='latin-1')

# Load the endpoint data
fpth = model_ws / f"{mp_namea}.mpend"
_fix_fortran_floats(fpth)
e = flopy.utils.EndpointFile(fpth)
e0 = e.get_alldata()

#Plot the endpoint data.
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mm = flopy.plot.PlotMapView(modelgrid=mg, ax=ax)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
cmap = mpl.colors.ListedColormap(
    [
        "r",
        "g",
    ]
)
v = mm.plot_array(ibd, cmap=cmap, edgecolor="gray")
mm.plot_endpoint(e0, direction="ending", colorbar=True, shrink=0.5)
plt.show()


# =============================================================================
# BTC ANALYSIS – travel times from MODPATH pathlines
# =============================================================================
fname_pth = model_ws / f"{mp_namea}.mppth"
_fix_fortran_floats(fname_pth)
p_btc = flopy.utils.PathlineFile(fname_pth)

b = float(top[0]) - float(botm[-1, 0])   # aquifer thickness (uniform)
porosity_val = 0.3  # must match Modpath7Bas porosity

arrival_times   = []
trace_lengths   = []
particle_velocities = []
ending_radii    = []
valid_particle_ids = []

for pid in range(p_btc.get_maxid() + 1):
    pi = p_btc.get_data(partid=pid)
    if len(pi) < 2:
        continue
    valid_particle_ids.append(pid)
    xp, yp = pi['x'], pi['y']
    seg_lengths  = np.sqrt(np.diff(xp)**2 + np.diff(yp)**2)
    total_length = float(np.sum(seg_lengths))
    total_time   = float(pi['time'][-1])
    ending_radii.append(float(np.sqrt((xp[-1] - cx)**2 + (yp[-1] - cy)**2)))
    arrival_times.append(total_time)
    trace_lengths.append(total_length)
    particle_velocities.append(total_length / total_time if total_time > 0 else np.nan)

arrival_times       = np.array(arrival_times)
trace_lengths       = np.array(trace_lengths)
particle_velocities = np.array(particle_velocities)
ending_radii        = np.array(ending_radii)

arrival_years = arrival_times / 365.25

# 1 Numerical BTC (cumulative fraction vs time)
sorted_times       = np.sort(arrival_years)
cumulative_fraction = np.arange(1, len(sorted_times) + 1) / len(sorted_times)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sorted_times, cumulative_fraction, 'k-', lw=2)
ax.set_xlabel("Travel time (years)")
ax.set_ylabel("Cumulative fraction of particles arrived")
ax.set_title("Numerical BTC at the Well (MODPATH)")
ax.set_xlim(left=0)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2 Analytical BTC – radial flow to a well (Darcy, piston)
#   t = π·b·n / |Q| · (R² – r_ring²)
t_analytical       = (np.pi * b * porosity_val / abs(-5000)) * (ending_radii**2 - r_ring**2)
t_analytical_years = t_analytical / 365.25

sorted_analytical   = np.sort(t_analytical_years)
cum_frac_analytical = np.arange(1, len(sorted_analytical) + 1) / len(sorted_analytical)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sorted_times,       cumulative_fraction,   'k-',  lw=2, label='Numerical (MODPATH)')
ax.plot(sorted_analytical,  cum_frac_analytical,   'r--', lw=2, label='Analytical (radial)')
ax.set_xlabel("Travel time (years)")
ax.set_ylabel("Cumulative fraction of particles arrived")
ax.set_title("BTC Comparison: Numerical vs Analytical")
ax.set_xlim(left=0)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Relative error per particle
rel_error = np.abs(arrival_times - t_analytical) / t_analytical * 100

# 3 Histogram of arrival times
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(arrival_years, bins=20, edgecolor='k', alpha=0.7)
ax.set_xlabel("Travel time (years)")
ax.set_ylabel("Number of particles")
ax.set_title("Distribution of Numerical Arrival Times (MODPATH)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4 Arrival times by particle ID
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(valid_particle_ids, arrival_years, color='mediumseagreen', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xlabel("Particle ID")
ax.set_ylabel("Arrival time (years)")
ax.set_title(f"Particle Arrival Times at the Well ({len(valid_particle_ids)}/{N_particles} particles)")
ax.grid(True, alpha=0.3)

# Add value labels on bars
for pid, time_years in zip(valid_particle_ids, arrival_years):
    ax.text(pid, time_years + max(arrival_years)*0.01, f'{time_years:.1f}', 
            ha='center', va='bottom', fontsize=8, rotation=0)

# Add summary statistics as text box
stats_text = (f'Valid particles: {len(valid_particle_ids)}/{N_particles}\n'
              f'Min: {arrival_years.min():.1f} years\n'
              f'Max: {arrival_years.max():.1f} years\n'
              f'Mean: {arrival_years.mean():.1f} years\n'
              f'Std: {arrival_years.std():.1f} years')
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# =============================================================================
# ALTERNATIVE PLOT: ALL 30 PARTICLES (INCLUDING INCOMPLETE JOURNEYS)
# =============================================================================

# Get travel times for ALL particles (including incomplete ones)
all_times = []
all_statuses = []
particle_ids_all = []
for pid in range(N_particles):
    pi = p_btc.get_data(partid=pid)
    if len(pi) > 0:
        final_time = float(pi['time'][-1])
        all_times.append(final_time)
        particle_ids_all.append(pid)
        
        # Determine status from endpoint data
        endpoint_mask = e0["particleid"] == pid
        if np.any(endpoint_mask):
            status = e0["status"][endpoint_mask][0]
            all_statuses.append(status)
        else:
            all_statuses.append(-1)  # Unknown status

all_times_years = np.array(all_times) / 365.25
all_statuses = np.array(all_statuses)

# Create color map for different statuses
colors = []
labels = []
for status in all_statuses:
    if status == 5:
        colors.append('green')
        labels.append('Completed (Well)')
    elif status == 9:
        colors.append('orange')  
        labels.append('Incomplete (Time limit)')
    else:
        colors.append('red')
        labels.append('Unknown')

# Plot all particles with status-based coloring
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(particle_ids_all, all_times_years, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

ax.set_xlabel("Particle ID")
ax.set_ylabel("Travel time (years)")
ax.set_title(f"All Particle Travel Times (Complete and Incomplete Journeys)")
ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.7, label=f'Completed at well ({np.sum(all_statuses == 5)} particles)'),
                   Patch(facecolor='orange', alpha=0.7, label=f'Hit time limit ({np.sum(all_statuses == 9)} particles)')]
if np.any(all_statuses == -1):
    legend_elements.append(Patch(facecolor='red', alpha=0.7, label=f'Unknown status ({np.sum(all_statuses == -1)} particles)'))

ax.legend(handles=legend_elements, loc='upper left')

# Add horizontal line at simulation time limit
ax.axhline(y=500000/365.25, color='red', linestyle='--', alpha=0.7, label='Simulation time limit')

# Add statistics box
completed = np.sum(all_statuses == 5)
incomplete = np.sum(all_statuses == 9)
completed_times = all_times_years[all_statuses == 5]
if len(completed_times) > 0:
    stats_text = (f'Total particles: {len(particle_ids_all)}\n'
                  f'Completed journeys: {completed}\n'
                  f'Incomplete journeys: {incomplete}\n'
                  f'Mean completion time: {completed_times.mean():.0f} years\n'
                  f'Simulation limit: {500000/365.25:.0f} years')
else:
    stats_text = f'Total particles: {len(particle_ids_all)}\nNo completed journeys'

ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# 5 Trace lengths and mean velocities
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(valid_particle_ids, trace_lengths, color='steelblue')
axes[0].set_xlabel("Particle ID")
axes[0].set_ylabel("Trace length (m)")
axes[0].set_title("Numerical Pathline Lengths (MODPATH)")
axes[1].bar(valid_particle_ids, particle_velocities, color='coral')
axes[1].set_xlabel("Particle ID")
axes[1].set_ylabel("Mean pore velocity (m/d)")
axes[1].set_title("Numerical Mean Pore Velocities (MODPATH)")
plt.tight_layout()
plt.show()


# 3D PYVISTA VISUALIZATION OF MODPATH PATHLINES
from flopy.utils import PathlineFile
from flopy.export.vtk import Vtk
import pyvista as pv
import numpy as np


# Load the pathlines produced by MODPATH
pathline_file = model_ws / f"{mp_namea}.mppth"
_fix_fortran_floats(pathline_file)
pf = PathlineFile(pathline_file)
pl = pf.get_alldata()


# Build VTK object from your DISV model and pathlines
vtk = Vtk(model=gwf, binary=False, vertical_exageration=20, smooth=False)
vtk.add_model(gwf)
vtk.add_pathline_points(pl)

# Convert to PyVista
grid, pathlines = vtk.to_pyvista()

# Derive a 3D point for the pumping well from the intersected DISV cell center.
wel_icpl = int(welcells["nodenumber"][0])
wel_k = int(welspd[0][0][0])
wel_x = float(mg.xcellcenters[wel_icpl])
wel_y = float(mg.ycellcenters[wel_icpl])
wel_z = float(mg.zcellcenters[wel_k, wel_icpl])
well_xyz = np.array(
    [[wel_x, wel_y, wel_z]]
)
well_point = pv.PolyData(well_xyz)

# Build a vertical well column so it is visible in 3D even when zoomed out.
well_top = float(top[wel_icpl])
well_bottom = float(botm[-1, wel_icpl])
well_line = pv.Line((wel_x, wel_y, well_top), (wel_x, wel_y, well_bottom))
well_tube = well_line.tube(radius=35.0)


# Static 3D plot
pv.set_plot_theme("document")

p = pv.Plotter()
p.add_mesh(grid, color="white", opacity=0.1)
p.add_mesh(
    pathlines,
    scalars="time",
    cmap="viridis",
    point_size=6,
    render_points_as_spheres=True,
)

p.add_axes()
p.camera.zoom(1.3)
p.show()


# Build particle tracks dictionary for animation
tracks = {}
particle_ids = set()
release_locs = []

times = pathlines["time"]
pids = pathlines["particleid"]
xyz = pathlines.points

for i in range(len(times)):
    pid = int(pids[i])
    loc = xyz[i]
    t = times[i]

    if pid not in tracks:
        tracks[pid] = []
        particle_ids.add(pid)
        release_locs.append(loc)

    tracks[pid].append((loc, t))

release_locs = np.array(release_locs)
tracks = {pid: np.array(track, dtype=object) for pid, track in tracks.items()}
max_track_len = max(len(track) for track in tracks.values())

# Explicitly release resources before cleaning temporary workspace on Windows.
if hasattr(hdobj, "close"):
    hdobj.close()
plt.close("all")
del hdobj
del grd
gc.collect()

try:
    temp_dir.cleanup()
except PermissionError as err:
    print(f"Warning: temporary workspace cleanup skipped due to file lock: {err}")