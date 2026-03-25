import os
import re
import numpy as np
import matplotlib.pyplot as plt
import flopy as fp
from flopy.utils.gridgen import Gridgen

modelname = 'webinar'
modelws = './webinar'
path = os.getcwd()

gridgen_exe = os.path.join(os.getcwd(), "gridgen_x64.exe")
plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams["figure.autolayout"] = True

# Model parameters
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

# Aquifer parameters
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

# Transport parameters
alpha_l          = 50.0      # longitudinal dispersivity (m)
alpha_th         = 5.0       # transverse horizontal dispersivity (m)
diffc            = 8.64e-5   # effective diffusion coefficient (mÂ²/d)
c_source         = 1.0       # input concentration at circular boundary (dimensionless)

# Time discretisation for transport
# Mean advective travel time from boundary to well: Ï€Â·RÂ²Â·nÂ·b / Q â‰ˆ 58 905 days
# Use ~1.7Ã— that to capture the full breakthrough curve
perlen_transport = 100_000.0   # total simulation time (days, ~273 years)
nstp_transport   = 200         # number of time steps
tsmult_transport = 1.03        # time-step multiplier (geometrically increasing)

mf6_exe = os.path.join(os.getcwd(), "mf6.exe")
mpath7_exe = os.path.join(os.getcwd(), "mpath7.exe")

# =============================================================================
# GRIDGEN: build refined DISV grid with nested zones around the well
# =============================================================================
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
#   level 1 Ã¢â€ â€™ cells ~25 m   (outer zone  Ã‚Â±1500 m)
#   level 2 Ã¢â€ â€™ cells ~12.5 m (middle zone  Ã‚Â±750 m)
#   level 3 Ã¢â€ â€™ cells  ~6.25 m (inner zone   Ã‚Â±300 m)
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

# =============================================================================
# Circular active domain mapped onto the DISV grid
# =============================================================================
dist_sq      = (xc_cells - circle_center_x)**2 + (yc_cells - circle_center_y)**2
active_circle = dist_sq <= circle_radius**2

# idomain for DISV: shape (nlay, ncpl) Ã¢â‚¬â€ 0 = inactive (outside circle)
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

# =============================================================================
# MF6 simulation
# =============================================================================
sim = fp.mf6.MFSimulation(sim_name=modelname,
                          version='mf6',
                          exe_name=mf6_exe,
                          sim_ws=modelws)

# Time discretization (transient – needed for transport)
tdis_rc = [(perlen_transport, nstp_transport, tsmult_transport)]
tdis = fp.mf6.ModflowTdis(sim, time_units="DAYS", perioddata=tdis_rc)

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

# Initial conditions Ã¢â‚¬â€ uniform head (h1 == h2 == 100)
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

# Check input Ã¢â‚¬â€ grid + BCs
pmv = fp.plot.PlotMapView(model=gwf)
pmv.plot_bc('CHD', color='blue')
pmv.plot_bc('WEL', plotAll=True, color='red')
pmv.plot_grid(colors='silver', lw=0.3)
plt.title("DISV Grid with Boundary Conditions")
plt.show()

# =============================================================================
# GWT â€“ Groundwater Transport model
# =============================================================================
gwt_name = modelname + '_gwt'
gwt = fp.mf6.ModflowGwt(sim,
                          modelname=gwt_name,
                          model_nam_file=f"{gwt_name}.nam")

# IMS packages – each model needs its own solver in a separate solution group
ims_gwf = fp.mf6.ModflowIms(sim, pname='imsgwf', filename=f'{modelname}.ims',
                             complexity='SIMPLE', inner_dvclose=1e-6)
sim.register_ims_package(ims_gwf, [modelname])
ims_gwt = fp.mf6.ModflowIms(sim, pname='imsgwt', filename=f'{gwt_name}.ims',
                             complexity='MODERATE', inner_dvclose=1e-6)
sim.register_ims_package(ims_gwt, [gwt_name])

# Same DISV grid as GWF
fp.mf6.ModflowGwtdisv(gwt,
                       length_units='METERS',
                       nlay=nlay,
                       ncpl=ncpl,
                       top=top_arr,
                       botm=botm_arr,
                       nvert=nvert,
                       vertices=vertices,
                       cell2d=cell2d,
                       idomain=idomain_disv)

# Initial concentration: C = 0 everywhere
fp.mf6.ModflowGwtic(gwt, strt=0.0)

# Mass storage and transfer
fp.mf6.ModflowGwtmst(gwt, porosity=porosity)

# Advection â€“ upstream weighting (stable for large Peclet numbers)
fp.mf6.ModflowGwtadv(gwt, scheme='UPSTREAM')

# Dispersion
fp.mf6.ModflowGwtdsp(gwt, alh=alpha_l, ath1=alpha_th, diffc=diffc)

# Constant concentration C = 1 at the circular boundary (tracer source)
cnc_spd = [[(0, int(icpl)), c_source] for icpl in edge_cells]
fp.mf6.ModflowGwtcnc(gwt, stress_period_data={0: cnc_spd})

# Output control â€“ save concentration at every time step (needed for BTC)
fp.mf6.ModflowGwtoc(gwt,
                     budget_filerecord=f"{gwt_name}.cbc",
                     concentration_filerecord=f"{gwt_name}.ucn",
                     saverecord=[('CONCENTRATION', 'ALL'), ('BUDGET', 'ALL')])

# Add SSM package to GWT model (required when GWF has boundary packages)
# CNC handles boundary concentration; no auxiliary sources needed.
fp.mf6.ModflowGwtssm(gwt)


# GWFâ€“GWT exchange: links flow solution to transport
fp.mf6.ModflowGwfgwt(sim,
                      exgtype='GWF6-GWT6',
                      exgmnamea=modelname,
                      exgmnameb=gwt_name,
                      filename=f"{modelname}.gwfgwt")

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
head  = hobj.get_data(totim=hobj.get_times()[-1])  # last time step (steady-state flow)
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
# MODPATH 7 Ã¢â‚¬â€ backward particle tracking from the well
# =============================================================================
# MODPATH 7 terminates particles that START inside a strong sink/source cell
# (the well cell is a strong source in backward time).
# Fix: place particles in a ring of DISV cells surrounding the well at a small
# radius, finding the closest active cell to each ring point.
N_particles = 30
r_ring = 100.0   # 100 m ring Ã¢â‚¬â€ outside the well cell, inside the refined zone
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

plt.title("Pathlines from the Well (Backward)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()

# =============================================================================
# Post-processing: Breakthrough Curve at the pumping well
# =============================================================================
ucn_path  = os.path.join(modelws, f"{gwt_name}.ucn")
cobj      = fp.utils.HeadFile(ucn_path, text='CONCENTRATION')
btc_times = np.array(cobj.get_times())

# Extract concentration at the well cell over time
conc_well = np.array([
    cobj.get_data(totim=t)[0, 0, well_cell_icpl]
    for t in btc_times
])

# Linear time axis
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(btc_times, conc_well / c_source, 'b-', lw=1.5)
ax.axvline(np.pi * circle_radius**2 * porosity * (top - botm[0]) / abs(Q_well),
           color='gray', ls='--', lw=1, label='Mean advective travel time')
ax.set_xlabel("Time (days)")
ax.set_ylabel("C / Câ‚€")
ax.set_title("Breakthrough Curve at Pumping Well")
ax.set_xlim(0, perlen_transport)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Log time axis â€“ better resolves early arrival and dispersion front
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(btc_times, conc_well / c_source, 'b-', lw=1.5)
ax.set_xlabel("Time (days, log scale)")
ax.set_ylabel("C / Câ‚€")
ax.set_title("Breakthrough Curve at Pumping Well (log-time)")
ax.set_ylim(0, 1.05)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()