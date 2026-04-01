import os
import re
import numpy as np
import matplotlib.pyplot as plt
import flopy as fp
from flopy.utils.gridgen import Gridgen

modelname = 'circular'
modelws = './circular'
path = os.getcwd()

gridgen_exe = os.path.join(os.getcwd(), "gridgen_x64.exe")
plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams["figure.autolayout"] = True

# MODEL PARAMETERS

# Grid parameters
Lx = 5000
Ly = 5000
ncol = 100
nrow = 100
delr = Lx / ncol   # 50 m
delc = Ly / nrow   # 50 m
print(f"delr: {delr}, delc: {delc}")

# Layer parameters (single layer, 50 m thick)
nlay = 1
top = 50
botm = [0]

# Aquifer parameters (homogeneous, isotropic)
k_x = 2.0
k_y = 2.0
k_z = 2.0
porosity = 0.3

# Circular active domain geometry
circle_center_x = Lx / 2.0
circle_center_y = Ly / 2.0
circle_radius = min(Lx, Ly) / 2.0

# Boundary conditions
Q_well = -5000
h1 = 100
h2 = 100

mf6_exe = os.path.join(os.getcwd(), "mf6.exe")
mpath7_exe = os.path.join(os.getcwd(), "mpath7.exe")

# GRIDGEN (build refined DISV grid with nested zones around the well)

# Base MODFLOW 2005 model used only to define the background grid for Gridgen
ms = fp.modflow.Modflow(exe_name=mf6_exe)
fp.modflow.ModflowDis(ms, nlay=nlay, nrow=nrow, ncol=ncol,
                      delr=delr, delc=delc, top=top, botm=botm[0])

gridgen_ws = os.path.join(modelws, 'gridgen')
g = Gridgen(ms.modelgrid, model_ws=gridgen_ws, exe_name=gridgen_exe)

# Helper: build a closed rectangular polygon (reversed for Gridgen orientation)
def _rect_poly(xmin, xmax, ymin, ymax):
    return [[list(reversed([
        (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)
    ]))]]

cx, cy = circle_center_x, circle_center_y

# Three nested refinement zones centred on the well (model centre)
#   level 1 at cells ~25 m   (outer zone  at 1500 m)
#   level 2 at cells ~12.5 m (middle zone  at 750 m)
#   level 3 at cells  ~6.25 m (inner zone   at 300 m)
g.add_refinement_features(_rect_poly(cx-1500, cx+1500, cy-1500, cy+1500),
                           "polygon", 1, range(nlay))
g.add_refinement_features(_rect_poly(cx-750,  cx+750,  cy-750,  cy+750),
                           "polygon", 2, range(nlay))
g.add_refinement_features(_rect_poly(cx-300,  cx+300,  cy-300,  cy+300),
                           "polygon", 3, range(nlay))

g.build(verbose=False)

# Extract DISV grid properties
gridprops = g.get_gridprops_disv()
ncpl      = gridprops["ncpl"]
top_arr   = gridprops["top"]
botm_arr  = gridprops["botm"]
nvert     = gridprops["nvert"]
vertices  = gridprops["vertices"]
cell2d    = gridprops["cell2d"]

print(f"Total cells in DISV grid: {ncpl}")

# Cell centres from cell2d: each entry is [icell2d, xc, yc, ncvert, icvert...]
xc_cells = np.array([c[1] for c in cell2d])
yc_cells = np.array([c[2] for c in cell2d])

# Circular active domain mapped onto the DISV grid
dist_sq      = (xc_cells - circle_center_x)**2 + (yc_cells - circle_center_y)**2
active_circle = dist_sq <= circle_radius**2

# idomain for DISV: shape (nlay, ncpl) â€” 0 = inactive (outside circle)
idomain_disv = np.zeros((nlay, ncpl), dtype=int)
idomain_disv[:, active_circle] = 1

# Circle edge cells: active cells within one original cell-width of the boundary
dist          = np.sqrt(dist_sq)
tolerance     = max(delr, delc)                    # 50 m
circle_edge_disv = active_circle & (dist >= circle_radius - tolerance)

# Well cell: active cell whose centre is closest to the model centre
dist_to_center = np.where(active_circle, dist_sq, np.inf)
well_cell_icpl = int(np.argmin(dist_to_center))
print(f"Well cell: icpl={well_cell_icpl}, "
      f"centre=({xc_cells[well_cell_icpl]:.1f}, {yc_cells[well_cell_icpl]:.1f})")


# MF6 SIMUALTION SETUP

sim = fp.mf6.MFSimulation(sim_name=modelname,
                          version='mf6',
                          exe_name=mf6_exe,
                          sim_ws=modelws)

# Time discretization
tdis_rc = [(1.0, 1, 1.0)]
tdis = fp.mf6.ModflowTdis(sim, time_units="DAYS", perioddata=tdis_rc)

# Iterative model solution
ims = fp.mf6.ModflowIms(sim, complexity='SIMPLE', inner_dvclose=1e-6)

# Groundwater flow model
gwf = fp.mf6.ModflowGwf(sim,
                         modelname=modelname,
                         model_nam_file=f"{modelname}.nam",
                         save_flows=True)

# DISV discretization (replaces DIS)
disv = fp.mf6.ModflowGwfdisv(gwf,
                              length_units='METERS',
                              nlay=nlay,
                              ncpl=ncpl,
                              top=top_arr,
                              botm=botm_arr,
                              nvert=nvert,
                              vertices=vertices,
                              cell2d=cell2d,
                              idomain=idomain_disv)

# Node property flow package
npf = fp.mf6.ModflowGwfnpf(gwf,
                            save_specific_discharge=True,
                            icelltype=1,
                            k=k_x,
                            k22=k_y,
                            k33=k_z)

# Initial conditions: uniform head (h1 == h2)
ic = fp.mf6.ModflowGwfic(gwf, strt=h1)

# Specified Head on the circular edge
chd_spd = []
edge_cells = np.where(circle_edge_disv)[0]
for k in range(nlay):
    for icpl in edge_cells:
        chd_spd.append([(k, int(icpl)), h1])

chd = fp.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)

# Well package
well_spd = [[(0, well_cell_icpl), Q_well]]
wel = fp.mf6.ModflowGwfwel(gwf, stress_period_data=well_spd)

# Output control
oc = fp.mf6.ModflowGwfoc(gwf,
                          head_filerecord=f"{modelname}.hds",
                          budget_filerecord=f"{modelname}.cbb",
                          saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],
                          printrecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')])

# Check input (grid + BCs)
pmv = fp.plot.PlotMapView(model=gwf)
pmv.plot_bc('CHD', color='blue')
pmv.plot_bc('WEL', plotAll=True, color='red')
pmv.plot_grid(colors='silver', lw=0.3)
plt.title("DISV Grid with Boundary Conditions")
plt.show()

# =============================================================================
# Run the model
# =============================================================================
sim.write_simulation()
sim.run_simulation()

# =============================================================================
# Post-processing: heads
# =============================================================================
fname = os.path.join(modelws, f'{modelname}.hds')
hobj  = fp.utils.HeadFile(fname)
head  = hobj.get_data(totim=1.0)       # shape (nlay, 1, ncpl) for DISV
head_lay1        = head[0, 0, :]       # shape (ncpl,)
head_lay1_masked = np.ma.masked_where(idomain_disv[0] == 0, head_lay1)
active_heads     = np.ma.masked_where(idomain_disv == 0, head[:, 0, :])
print(f"max head: {np.ma.max(active_heads):.3f}, "
      f"min head: {np.ma.min(active_heads):.3f}")

h_min = float(np.ma.min(head_lay1_masked))
h_max = float(np.ma.max(head_lay1_masked))

# Contour plot
pmv = fp.plot.PlotMapView(model=gwf)
qm  = pmv.plot_array(head_lay1_masked)
plt.colorbar(qm, shrink=0.5, label='Head (m)')
cs = pmv.contour_array(head_lay1_masked,
                       levels=np.linspace(h_min, h_max, 15),
                       linewidths=1, colors='k')
plt.clabel(cs, fmt='%1.1f')
plt.title("Head Contours for Layer 1")
plt.show()

# =============================================================================
# Post-processing: flow vectors
# =============================================================================
fname = os.path.join(modelws, f"{modelname}.cbb")
cbb   = fp.utils.CellBudgetFile(fname, precision='double')
spdis = cbb.get_data(text='DATA-SPDIS')[0]

qx = np.full(ncpl, np.nan, dtype=float)
qy = np.full(ncpl, np.nan, dtype=float)
active_nodes = np.where(idomain_disv[0] > 0)[0]
qx[active_nodes] = spdis["qx"]
qy[active_nodes] = spdis["qy"]

# VELOCITIES 
vx = np.full(ncpl, np.nan, dtype=float)
vy = np.full(ncpl, np.nan, dtype=float)
vx[active_nodes] = qx[active_nodes]/porosity 
vy[active_nodes] = qy[active_nodes]/porosity


fig, ax = plt.subplots(figsize=(7, 7))
pmv = fp.plot.PlotMapView(model=gwf, ax=ax)
pmv.plot_vector(qx, qy, scale=1.0, color="black")
pmv.plot_grid(colors="0.75", lw=0.3)
ax.set_aspect("equal", adjustable="box")
ax.set_xlim((0, Lx))
ax.set_ylim((0, Ly))
ax.set_title("Flow Vectors")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
plt.show()

# =============================================================================
# MODPATH 7 â€” backward particle tracking from the well
# =============================================================================
# MODPATH 7 terminates particles that START inside a strong sink/source cell
# (the well cell is a strong source in backward time).
# Fix: place particles in a ring of DISV cells surrounding the well at a small
# radius, finding the closest active cell to each ring point.
N_particles = 30
r_ring = 50.0   # 100 m ring â€” outside the well cell, inside the refined zone
theta = np.linspace(0, 2 * np.pi, N_particles, endpoint=False)
ring_x = cx + r_ring * np.cos(theta)
ring_y = cy + r_ring * np.sin(theta)

# For each ring point find the closest active DISV cell (excluding the well cell)
plocs = []
for px, py in zip(ring_x, ring_y):
    d = (xc_cells - px) ** 2 + (yc_cells - py) ** 2
    d[~active_circle] = np.inf
    d[well_cell_icpl] = np.inf          # exclude the strong sink cell itself
    plocs.append(int(np.argmin(d)))     # absolute node number (layer 0)

# All particles start at the cell centre (local coords = 0.5)
localx = [0.5] * N_particles
localy = [0.5] * N_particles
localz = [0.5] * N_particles

particledata = fp.modpath.ParticleData(plocs,
                                       structured=False,
                                       localx=localx,
                                       localy=localy,
                                       localz=localz)

pg = fp.modpath.ParticleGroup(particledata=particledata)

mp = fp.modpath.Modpath7(modelname=f"{modelname}_abs_wells_bw",
                          model_ws=modelws,
                          flowmodel=gwf,
                          exe_name=mpath7_exe)

mpbas = fp.modpath.Modpath7Bas(mp, porosity=porosity)

mpsim = fp.modpath.Modpath7Sim(mp,
                                particlegroups=pg,
                                weaksourceoption="pass_through",
                                stoptime=100 * 365.,
                                trackingdirection='backward')

mp.write_input()
mp.run_model()

# Fix malformed scientific notation in MODPATH output
mppth_path = os.path.join(modelws, f"{modelname}_abs_wells_bw.mppth")
with open(mppth_path, 'r') as f:
    content = f.read()
fixed = re.sub(r'(\d)([+-]\d{3})', r'\1E\2', content)
with open(mppth_path, 'w') as f:
    f.write(fixed)

# Read and plot pathlines
pmv = fp.plot.PlotMapView(model=gwf)
pmv.contour_array(head_lay1_masked,
                  levels=np.linspace(h_min, h_max, 15),
                  colors='C0')
p = fp.utils.PathlineFile(mppth_path)
for i in range(p.get_maxid() + 1):
    pi = p.get_data(partid=i)
    plt.plot(pi['x'], pi['y'], 'C1', lw=1)
    plt.text(pi['x'][-1], pi['y'][-1], str(i), fontsize=7, ha='center', va='bottom')

plt.title("Pathlines from the Well (Backward)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()


# Numerical BTC from MODPATH

# MODPATH already computes travel times along each pathline using
# the velocity field and porosity. For backward tracking the
# recorded time at the last waypoint of each particle equals
# the total travel time from its recharge location to the well.

# 1 Extract arrival times and trace lengths per particle 
arrival_times = []   # total travel time per particle [days]
trace_lengths = []   # total path length per particle  [m]
particle_velocities = []  # mean pore velocity per particle [m/d]
ending_radii = []    # radial distance of last waypoint from well [m]

fname = os.path.join(modelws, f"{modelname}_abs_wells_bw.mppth")
p = fp.utils.PathlineFile(fname)

# Loop over all particles and extract their pathline data to compute total travel time, path length, and mean velocity.
# Each particle's pathline is represented as a sequence of waypoints (MODPATH records a new waypoint every time a particle crosses a cell boundary) 
# with coordinates and travel times. 
# We compute the total path length by summing the distances between consecutive waypoints. 
# The total travel time is given by the time at the last waypoint. 
# The mean pore velocity is then calculated as total path length divided by total travel time.

for pid in range(p.get_maxid() + 1): # loop over all particle IDs
    pi = p.get_data(partid=pid) # get pathline data for particle with ID 'pid' (returns a structured array with fields like 'x', 'y', 'z', 'time', etc.)
    if len(pi) < 2: # skip particles with less than 2 waypoints (no meaningful pathline)
        continue

    # Coordinates along the pathline
    xp = pi['x'] # x-coordinates of the pathline waypoints
    yp = pi['y'] # y-coordinates of the pathline waypoints

    # Segment lengths and cumulative path length
    dx = np.diff(xp) # differences in x-coordinates between consecutive waypoints
    dy = np.diff(yp) # differences in y-coordinates between consecutive waypoints
    seg_lengths = np.sqrt(dx**2 + dy**2) # Euclidean distances between consecutive waypoints
    total_length = np.sum(seg_lengths) # total pathline length

    # Travel time recorded by MODPATH (last waypoint)
    total_time = pi['time'][-1]          # [days]

    # Ending radius: distance from well centre to last waypoint (boundary location)
    r_end = np.sqrt((xp[-1] - circle_center_x)**2 + (yp[-1] - circle_center_y)**2)
    ending_radii.append(r_end)

    arrival_times.append(total_time) # append total travel time for this particle to the list
    trace_lengths.append(total_length) # append total path length for this particle to the list
    if total_time > 0: # compute mean pore velocity as total path length divided by total travel time, and append to the list
        particle_velocities.append(total_length / total_time)  # [m/d]
    else:
        particle_velocities.append(np.nan)

arrival_times = np.array(arrival_times) # convert lists to numpy arrays for easier analysis and plotting
trace_lengths = np.array(trace_lengths) # convert lists to numpy arrays for easier analysis and plotting
particle_velocities = np.array(particle_velocities) # convert lists to numpy arrays for easier analysis and plotting
ending_radii = np.array(ending_radii)

# Print summary
print(f"\nNumber of particles : {len(arrival_times)}") 
print(f"Arrival times [years]: min={arrival_times.min()/365.25:.1f}, "
      f"max={arrival_times.max()/365.25:.1f}, mean={arrival_times.mean()/365.25:.1f}")
print(f"Trace lengths [m]  : min={trace_lengths.min():.1f}, "
      f"max={trace_lengths.max():.1f}, mean={trace_lengths.mean():.1f}")
print(f"Mean velocities [m/d]: min={np.nanmin(particle_velocities):.4f}, "
      f"max={np.nanmax(particle_velocities):.4f}")

# Convert to years for readability
arrival_years = arrival_times / 365.25

# 2 Breakthrough curve (cumulative fraction vs time) 
sorted_times = np.sort(arrival_years) # sort the arrival times in ascending order, from smallest to largest, to prepare for plotting the breakthrough curve (cumulative fraction of particles arrived vs travel time)
cumulative_fraction = np.arange(1, len(sorted_times) + 1) / len(sorted_times) # compute the cumulative fraction of particles arrived at the well as a function of travel time (sorted_times).

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


# Analytical BTC radial flow to a well in a circular confined (?) aquifer
b = top - botm[0]  # aquifer thickness [m]

# Per-particle analytical travel time using each particle's actual ending radius
t_analytical = (np.pi * b * porosity / abs(Q_well)) * (ending_radii**2 - r_ring**2)  # [days]
t_analytical_years = t_analytical / 365.25

# Analytical BTC (cumulative fraction vs time)
sorted_analytical = np.sort(t_analytical_years)
cum_frac_analytical = np.arange(1, len(sorted_analytical) + 1) / len(sorted_analytical)

print(f"\nAnalytical travel times [years]: "
      f"min={t_analytical_years.min():.2f}, "
      f"max={t_analytical_years.max():.2f}, "
      f"mean={t_analytical_years.mean():.2f}")

# Comparison plot: numerical vs analytical BTC
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sorted_times, cumulative_fraction, 'k-', lw=2, label='Numerical (MODPATH)')
ax.plot(sorted_analytical, cum_frac_analytical, 'r--', lw=2, label='Analytical (radial)')
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
print(f"Relative error [%]: min={rel_error.min():.2f}, "
      f"max={rel_error.max():.2f}, mean={rel_error.mean():.2f}")

# 3 Histogram of arrival times 
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(arrival_years, bins=20, edgecolor='k', alpha=0.7)
ax.set_xlabel("Travel time (years)")
ax.set_ylabel("Number of particles")
ax.set_title("Distribution of Numerical Arrival Times (MODPATH)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4 Trace lengths and velocities summary plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(len(trace_lengths)), trace_lengths, color='steelblue')
axes[0].set_xlabel("Particle ID")
axes[0].set_ylabel("Trace length (m)")
axes[0].set_title("Numerical Pathline Lengths (MODPATH)")

axes[1].bar(range(len(particle_velocities)), particle_velocities, color='coral')
axes[1].set_xlabel("Particle ID")
axes[1].set_ylabel("Mean pore velocity (m/d)")
axes[1].set_title("Numerical Mean Pore Velocities (MODPATH)")

plt.tight_layout()
plt.show()