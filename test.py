## Creating a Simple MODFLOW 6 Model with Flopy (DISV - Refined Grid)
 
# This script builds a MODFLOW 6 model using a Gridgen-refined DISV grid.
# The refined mesh is used for both the simulation and all result plots.
 
# Setup the Notebook Environment
from flopy.export.vtk import Vtk
import os
import sys
from pprint import pformat
import tempfile
 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
 
import pyvista as pv
 
import flopy
 
from shapely.geometry import Polygon
from flopy.utils.gridgen import Gridgen
 
print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")
 
# For this example, we will set up a temporary workspace.
workspace = os.path.join(tempfile.gettempdir(), "cube")
 
# Model Parameters
name = "cube"
h1 = 100    # constant head value for boundaries (except lake)
h2 = 90     # constant head value 
Nlay = 10
N = 10      # number of rows and columns
L = 100.0   # length of the model in the x and y directions
H = 100.0   # aquifer thickness
k_x = 100.0 # horizontal hydraulic conductivity - x direction
k_y = 10.0   # horizontal hydraulic conductivity - y direction
k_z = 100.0 # vertical hydraulic conductivity - z direction (base value, halves per layer)
bot = np.linspace(-H / Nlay, -H, Nlay) # layer bottom elevations
delrow = delcol = L / (N - 1)
 
# Boundary conditions: True = constant head, False = no-flow
chd_left = True   # left boundary (x = 0)
chd_right = True  # right boundary (x = L)
chd_top = False    # top boundary (y = L)
chd_bottom = False # bottom boundary (y = 0)
 
# Lake location (x, y coordinates in model units)
lake_x = 50.0 # lake x position
lake_y = 50.0 # lake y position
lake_layers = 1 # number of layers the lake occupies (from top)
 
# Well locations (x, y coordinates in model units)
well_x = 30.0 # well x position
well_y = 30.0 # well y position
well_layer = 0 # layer the well is in (0 = top)
well_rate = -1000.0 # pumping rate (negative = extraction)
 
# DIS - Create a base structured grid for Gridgen
# A temporary DIS model is needed as the base grid for Gridgen refinement
sim_base = flopy.mf6.MFSimulation(sim_name="base", sim_ws=workspace)
gwf_base = flopy.mf6.ModflowGwf(sim_base, modelname="base")
dis_base = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(gwf_base,
                                                    nlay=Nlay,
                                                    nrow=N,
                                                    ncol=N,
                                                    delr=delrow,
                                                    delc=delcol,
                                                    top=0.0,
                                                    botm=bot,)
 
# Gridgen
g = Gridgen(gwf_base.modelgrid, model_ws=name, exe_name="gridgen_x64")
 
center_poly = [Polygon([(40, 40), (60, 40), (60, 60), (40, 60)])]
g.add_refinement_features(center_poly, "polygon", 2, list(range(Nlay)))
g.build()
 
# Plot the refined grid from Gridgen
vtk_path = os.path.join(name, "qtg_sv.vtu") # standard Gridgen output file: quadtree grid surface vertices in VTK Unstructured format
if os.path.exists(vtk_path):
    refined_grid = pv.read(vtk_path)
    p = pv.Plotter()
    p.add_mesh(refined_grid, show_edges=True, opacity=1)
    p.add_axes()
    p.show(title="3D Refined Grid from Gridgen")
else:
    print(f"VTK file not found: {vtk_path}")
 
# Get DISV properties from Gridgen
gridprops = g.get_gridprops_disv()
ncpl = gridprops["ncpl"]
 
# Create the MODFLOW 6 simulation with DISV - DISV
sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=workspace)
 
tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
    sim, pname="tdis", time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)]
)
 
model_nam_file = f"{name}.nam"
gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file)
 
ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", complexity="SIMPLE")
 
# Create DISV package from Gridgen output
disv = flopy.mf6.ModflowGwfdisv(gwf, pname="disv", **gridprops)
 
# Get cell centers from the DISV modelgrid
mg = gwf.modelgrid
xc = np.array(mg.xcellcenters).flatten()
yc = np.array(mg.ycellcenters).flatten()
 
# Grid extents
x_extent = N * delcol
y_extent = N * delrow
 
# Create model packages for the DISV grid
 
# IC - Initial conditions
start = h1 * np.ones((Nlay, ncpl)) # ncpl = number of cells per layer
ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=start)
 
# NPF - Node Property Flow (conductivities)
k_arr = np.full((Nlay, ncpl), k_x) # horizontal hydraulic conductivity - x
k22_arr = np.full((Nlay, ncpl), k_y) # horizontal hydraulic conductivity - y
k33_arr = np.array([np.full(ncpl, k_z / (2 ** layer)) # vertical hydraulic conductivity - z (halves per layer)
                    for layer in range(Nlay)])
 
npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
    gwf, pname="npf", icelltype=[1] * Nlay,
    k=k_arr, k22=k22_arr, k33=k33_arr, save_flows=True,
)
 
# CHD - Constant Head boundaries
# For DISV, cellid is (layer, cell_in_layer) instead of (layer, row, col)
chd_rec = []
 
# Boundary cells: cells whose centers are within one base cell width of the grid edges
for icell in range(ncpl):
    on_left = chd_left and xc[icell] < delcol
    on_right = chd_right and xc[icell] > x_extent - delcol
    on_bottom = chd_bottom and yc[icell] < delrow
    on_top = chd_top and yc[icell] > y_extent - delrow
    if on_left or on_right or on_bottom or on_top:
        for layer in range(Nlay):
            chd_rec.append(((layer, icell), h1))
 
# Lake
dists = np.sqrt((xc - lake_x)**2 + (yc - lake_y)**2)
lake_cell = int(np.argmin(dists))
for layer in range(lake_layers):
    chd_rec.append(((layer, lake_cell), h2))
 
chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(
    gwf, pname="chd", maxbound=len(chd_rec),
    stress_period_data=chd_rec, save_flows=True,
)
 
# ibd array for boundary visualization
ibd = np.ones((Nlay, ncpl), dtype=int)
ra = chd.stress_period_data.get_data(key=0)
for lay, icell in ra["cellid"]:
    ibd[lay, icell] = -1
 
# WELL
dists_wel = np.sqrt((xc - well_x)**2 + (yc - well_y)**2)
well_cell = int(np.argmin(dists_wel))
wel_rec = [((well_layer, well_cell), well_rate)]
wel = flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(gwf, stress_period_data=wel_rec)
 
for (lay, icell), q in wel_rec:
    ibd[lay, icell] = -2
 
# OC - Output Control
headfile = f"{name}.hds"
budgetfile = f"{name}.cbb"
oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(
    gwf, pname="oc",
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    head_filerecord=[headfile],
    budget_filerecord=[budgetfile],
    printrecord=[("HEAD", "LAST")],
)
 
 
# Build VTK, run simulation, and plot results
# Build PyVista grid from the DISV model (refined mesh)
vtk = Vtk(model=gwf, binary=False, smooth=False)
vtk.add_model(gwf)
grid = vtk.to_pyvista()
 
# Write and run the simulation
sim.write_simulation()
print(os.listdir(workspace))
 
success, buff = sim.run_simulation(silent=True, report=True)
assert success, pformat(buff)
 
# Read the binary head file and plot the results
fname = os.path.join(workspace, headfile)
hds = flopy.utils.binaryfile.HeadFile(fname)
h = hds.get_data(kstpkper=(0, 0))
 
# Add the head values as a dataset to the grid
grid["head"] = h.flatten()
grid["k_arr"] = npf.k.array.flatten()
grid["k22_arr"] = npf.k22.array.flatten()
grid["k33_arr"] = npf.k33.array.flatten()
grid["ibd"] = ibd.flatten()
 
# Plot all scalars on the refined mesh
scalars = ["head", "k_arr", "k22_arr", "k33_arr", "ibd"]
cmaps = ["viridis", "plasma", "plasma", "plasma", "coolwarm"]
for i, s in enumerate(scalars):
    p = pv.Plotter()
    p.add_mesh(grid, scalars=s, cmap=cmaps[i], show_edges=True, opacity=1)
    p.add_axes()
    p.show(title=s)