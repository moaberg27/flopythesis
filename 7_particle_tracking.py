# Set up the Environment
import os
import sys
import gc
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

proj_root = Path.cwd().parent.parent
script_dir = Path(__file__).resolve().parent


# Use executables from the FLOPY root directory
flopy_root = Path(__file__).resolve().parents[1]
mf_exe = (flopy_root / "mf6.exe").resolve()
gridgen_exe = (flopy_root / "gridgen_x64.exe").resolve()
mp_exe = (flopy_root / "mpath7.exe").resolve()

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

print(sys.version)
print("numpy version: {}".format(np.__version__))
print("matplotlib version: {}".format(mpl.__version__))
print("flopy version: {}".format(flopy.__version__))

# temporary directory
temp_dir = TemporaryDirectory(ignore_cleanup_errors=True)
workspace = Path(temp_dir.name)

# Creation of the model grid
Lx = 10000.0
Ly = 10000.0
nlay = 3
nrow = 20
ncol = 20
delr = Lx / ncol
delc = Ly / nrow
top = 400
botm = [250, 200, 0]
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
print(f"nlay: {nlay}, nrow: {nrow}, ncol: {ncol}")
print(f"delr: {delr}, delc: {delc}")

# Create the Gridgen object
from flopy.utils.gridgen import Gridgen

model_name = "mp7_3D"
model_ws = workspace / "mp7_3D" / "mf6"
gridgen_ws = model_ws / "gridgen"
g = Gridgen(ms.modelgrid, model_ws=gridgen_ws, exe_name=str(gridgen_exe))

# Refine the grid
rf0shp = gridgen_ws / "rf0"
xmin = 7 * delr
xmax = 12 * delr
ymin = 8 * delc
ymax = 13 * delc
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
xmin = 8 * delr
xmax = 11 * delr
ymin = 9 * delc
ymax = 12 * delc
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
xmin = 9 * delr
xmax = 10 * delr
ymin = 10 * delc
ymax = 11 * delc
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
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=300.0)

# node property flow
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    xt3doptions=[("xt3d")],
    icelltype=[1, 0, 0], # layer 1 is convertible, layers 2 and 3 are confined
    k=[50.0, 0.01, 200.0], # horizontal conductivity for layers 1, 2, and 3
    k33=[10.0, 0.01, 20.0], # vertical conductivity for layers 1, 2, and 3
)

# well
wellpoints = [(4750.0, 5250.0)]
welcells = g.intersect(wellpoints, "point", 0)

# welspd = flopy.mf6.ModflowGwfwel.stress_period_data.empty(gwf, maxbound=1, aux_vars=['iface'])
welspd = [[(2, icpl), -1000, 0] for icpl in welcells["nodenumber"]]
wel = flopy.mf6.ModflowGwfwel(
    gwf, print_input=True, auxiliary=[("iface",)], stress_period_data=welspd
)

# constant head boundary conditions on all 4 sides
h1 = 300.0  # left   (x = 0)
h2 = 300.0  # right  (x = Lx)
#h3 = 320.0  # bottom (y = 0)
#h4 = 320.0  # top    (y = Ly)

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
    #for icpl in bottom_cells:
        #if (lay, icpl) not in seen:
            #seen.add((lay, icpl))
            #chdspd.append([(lay, icpl), h3])
    #for icpl in top_cells:
        #if (lay, icpl) not in seen:
            #seen.add((lay, icpl))
            #chdspd.append([(lay, icpl), h4])

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

# Plot the head distribution in layer 3
ilay = 2
cint = 0.25
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mm = flopy.plot.PlotMapView(modelgrid=mg, ax=ax, layer=ilay)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
pc = mm.plot_array(head[:, 0, :], cmap="jet", edgecolor="black")
hmin = head[ilay, 0, :].min()
hmax = head[ilay, 0, :].max()
levels = np.arange(np.floor(hmin), np.ceil(hmax) + cint, cint)
cs = mm.contour_array(head[:, 0, :], colors="white", levels=levels)
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

pcoord = np.array(
    [
        [0.000, 0.125, 0.500],
        [0.000, 0.375, 0.500], 
        [0.000, 0.625, 0.500], 
        [0.000, 0.875, 0.500],
        [1.000, 0.125, 0.500],
        [1.000, 0.375, 0.500],
        [1.000, 0.625, 0.500],
        [1.000, 0.875, 0.500],
        [0.125, 0.000, 0.500],
        [0.375, 0.000, 0.500],
        [0.625, 0.000, 0.500],
        [0.875, 0.000, 0.500],
        [0.125, 1.000, 0.500],
        [0.375, 1.000, 0.500],
        [0.625, 1.000, 0.500],
        [0.875, 1.000, 0.500],
    ]
)
nodew = gwf.disv.ncpl.array * 2 + welcells["nodenumber"][0]
plocs = [nodew for i in range(pcoord.shape[0])]

# create particle data
pa = flopy.modpath.ParticleData(
    plocs,
    structured=False,
    localx=pcoord[:, 0],
    localy=pcoord[:, 1],
    localz=pcoord[:, 2],
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
flopy.modpath.Modpath7Bas(mp, porosity=0.1)
flopy.modpath.Modpath7Sim(
    mp,
    simulationtype="combined",
    trackingdirection="backward",
    weaksinkoption="pass_through",
    weaksourceoption="pass_through",
    referencetime=0.0,
    stoptimeoption="extend",
    timepointdata=[500, 1000.0],
    particlegroups=pgb,
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
    raw = fpath.read_text()
    fixed = re.sub(r'(\d)(\-)(\d)', r'\1E\2\3', raw)
    fixed = re.sub(r'(\d)(\+)(\d)', r'\1E\2\3', fixed)
    fpath.write_text(fixed)

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


# 3D PYVISTA VISUALIZATION OF MODPATH PATHLINES
print("Loading pathlines for 3D visualization...")

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

print("PyVista meshes created:")
print(" - Grid cells:", grid.n_cells)
print(" - Pathline points:", pathlines.n_points)

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
p.add_mesh(
    well_point,
    color="red",
    point_size=18,
    render_points_as_spheres=True,
)
p.add_mesh(well_tube, color="red", opacity=0.9)

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

print("Longest track length:", max_track_len)


# Animated GIF of particle motion
gif_path = model_ws / f"{model_name}_pathlines.gif"

p = pv.Plotter(off_screen=True)
try:
    p.open_gif(str(gif_path))
except ModuleNotFoundError:
    p.close()
    print("Skipping GIF export: install imageio with `pip install imageio`.")
    gif_path = None

# Particle points (moving)
moving_pts = pv.PolyData(release_locs)
moving_pts.point_data["time"] = np.zeros(len(release_locs))

p.add_mesh(grid, color="white", opacity=0.1)
p.add_mesh(moving_pts, cmap="viridis", clim=[0, times.max()])
p.add_mesh(well_tube, color="red", opacity=0.9)
p.add_mesh(
    well_point,
    color="red",
    point_size=18,
    render_points_as_spheres=True,
)
p.camera.zoom(1.3)

if gif_path is not None:
    for step in range(max_track_len):
        pts = []
        tvals = []

        for pid in particle_ids:
            tr = tracks[pid]
            if step < len(tr):
                loc, t = tr[step]
            else:
                loc, t = tr[-1]
            pts.append(loc)
            tvals.append(t)

        moving_pts.points = np.vstack(pts)
        moving_pts.point_data["time"] = np.array(tvals)

        p.write_frame()

    p.close()
    print("GIF saved to:", gif_path)

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