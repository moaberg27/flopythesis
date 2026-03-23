import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import flopy as fp

modelname = 'webinar'
modelws = './webinar'
path = os.getcwd()
plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams["figure.autolayout"] = True

# Model parameters
# Grid parameters
Lx = 5000
Ly = 5000
ncol = 100
nrow = 100
delr = Lx / ncol
delc = Ly / nrow
print(f"delr: {delr}, delc: {delc}")

# Layer parameters
nlay = 3
top = 100
botm = [60, 40, 0]

# Aquifer parameters
k_h = [20, 0.01, 10]
k_v = [20, 0.01, 10]
porosity = 0.3


# Boundary conditions
Q_well = -3000
h1 = 100
h2 = 100
# Create the Simulation
exe_name = "mf6.exe"
sim = fp.mf6.MFSimulation(sim_name=modelname, 
                          version='mf6',
                          exe_name=exe_name,
                          sim_ws=modelws,
                          )

# Time discretization
tdis_rc = [(1.0, 1, 1.0)]
tdis = fp.mf6.ModflowTdis(
    sim,
    time_units="DAYS",
    perioddata=tdis_rc,
)

# Iterative model solution
ims = fp.mf6.ModflowIms(sim, 
                        complexity='SIMPLE',
                        inner_dvclose=1e-6,
)

# Groundwater Flow Model
gwf = fp.mf6.ModflowGwf(sim, 
                        modelname=modelname, 
                        model_nam_file=f"{modelname}.nam",
                        save_flows=True)

# ADD PACKAGES 
# Discretization
dis = fp.mf6.ModflowGwfdis(gwf,
                           length_units='METERS', 
                           nlay=nlay, 
                           nrow=nrow, 
                           ncol=ncol, 
                           delr=delr, 
                           delc=delc,
                           top=top, 
                           botm=botm,
                           )

# Node property flow package
npf = fp.mf6.ModflowGwfnpf(gwf, 
                           save_specific_discharge=True,
                           icelltype=[1, 0, 0],
                           k = k_h, 
                           k33 = k_v,
                           )

# Initial Conditions
strt = np.full((nlay, nrow, ncol), (h1 + h2) / 2.0, dtype=float)
strt[:, :, 0] = h1
strt[:, :, -1] = h2

ic = fp.mf6.ModflowGwfic(gwf, 
                         strt=strt,
                         )

# Specified Head Package
chd_spd = []
for k in range(nlay):
    for r in range(nrow):
        chd_spd.append([(k, r, 0), h1])
        chd_spd.append([(k, r, ncol - 1), h2])

chd = fp.mf6.ModflowGwfchd(gwf, 
                           stress_period_data=chd_spd,
)


# Well Package
center_row = nrow // 2
center_col = ncol // 2
well_row_col = [(center_row, center_col)]
well_spd = [[(2, center_row, center_col), Q_well]]

wel = fp.mf6.ModflowGwfwel(gwf,
                        stress_period_data=well_spd,
)

# Output control
oc = fp.mf6.ModflowGwfoc(gwf,
                         head_filerecord=f"{modelname}.hds",
                         budget_filerecord=f"{modelname}.cbb",
                         saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],
                         printrecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')]
)

# Check input
pmv = fp.plot.PlotMapView(model=gwf)
pmv.plot_bc('CHD', color='blue')
pmv.plot_bc('WEL', plotAll=True, color='red')
pmv.plot_grid(colors='silver', lw=1)


plt.figure(figsize=(5, 3))
pxs = fp.plot.PlotCrossSection(model=gwf, line={'row': 50})
pxs.plot_bc('CHD', color='blue')
pxs.plot_bc('WEL', color='red')
pxs.plot_grid(colors='silver', lw=1)

plt.show()

# Solve the model
sim.write_simulation()
sim.run_simulation()

# Read Heads
fname = os.path.join(modelws, f'{modelname}.hds')
hobj = fp.utils.HeadFile(fname)
head = hobj.get_data(totim=1.0) # Get the head data for the first time step (totim=1.0) and store it in the variable 'head' as a 3D array (nlay, nrow, ncol).
head_lay3 = hobj.get_data(mflay=2) # Extract the head data for layer 3 and store it in the variable 'head_lay3' as a 2D array (nrow, ncol).
print(f"max head: {np.max(head)}, min head: {np.min(head)}")

# Contour Plot
pmv = fp.plot.PlotMapView(model=gwf)
qm = pmv.plot_array(head_lay3)
plt.colorbar(qm, shrink=0.5, label='Head (m)')
cs = pmv.contour_array(head_lay3, levels=range(80, 100), linewidths=1, colors='k') 
plt.clabel(cs, fmt='%1.0f')
plt.title("Head Contours for Layer 3")

plt.show()

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(ncol) * delr
y = np.arange(nrow) * delc
X, Y = np.meshgrid(x, y)

surf = ax.plot_surface(X, Y, head_lay3, cmap='viridis', linewidth=0, antialiased=True)
fig.colorbar(surf, shrink=0.6, label='Head (m)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Head (m)')
ax.set_title('3D Head Surface (Layer 3)')
plt.show()

# Water Balance 
fname = os.path.join(modelws, f"{modelname}.cbb")
cbb = fp.utils.CellBudgetFile(fname, precision='double')

# Read Flow and Quiver Plot
spdis = cbb.get_data(text='DATA-SPDIS')[0] 
qx = spdis["qx"].reshape((nlay, nrow, ncol))
qy = spdis["qy"].reshape((nlay, nrow, ncol)) 

for l in range(nlay):
    ax = plt.subplot(2, 2, l+1)
    pmv = fp.plot.PlotMapView(model=gwf, ax=ax, layer=l)
    pmv.plot_vector(qx[l], 
                        qy[l], 
                        scale=1,
                        istep=4,
                        jstep=4,
                        color="k",
                        )
    ax.set_xlim((0, Lx))
    ax.set_ylim((0, Ly))
    plt.title(f"Layer {l + 1}")

plt.tight_layout()
plt.show()

# Function to create N particle positions on a circle​
def PointsOnCircle(r=10, N=30): # default is 10 m radius and 28 particles​
    rv = [2 * np.pi /N *n_i for n_i in range(0,N)] # N angles equally spaced around the circle​
    return np.vstack((r * np.cos(rv), r * np.sin(rv))).T # convert to x and y coordinates and transpose to get N rows of (x,y) pairs​.

pr = PointsOnCircle()

# Define particle starting points inside well cells​

localx = []  # particle local coordinates inside MF cell​
localy = []
localz = []
partlocs = [] # list of (layer,row,col) for each particle​

for rc in well_row_col: # loop over all well cells​
    r = rc[0] # row index of the well cell​
    c = rc[1] # column index of the well cell​
    for pr_i in pr: # pr_i is one (x,y) coordinate from the circle around the well.​
        lx = (delc / 2 + pr_i[0]) / delc # compute local x coordinate of the particle in the cell.​
        ly = (delr / 2 + pr_i[1]) / delr
        lz = 0.5 # vertical position halfway in the layer​
        partlocs.append((2,r,c)) # particles are in layer 2, row r,column c.
        localx.append(lx)
        localy.append(ly)
        localz.append(lz)

particledata = fp.modpath.ParticleData(partlocs, 
                                       structured=True,
                                       localx=localx,
                                       localy=localy,
                                       localz=localz)

pg = fp.modpath.ParticleGroup(particledata=particledata) #create a particle group with the particle data​

mp = fp.modpath.Modpath7(modelname=f"{modelname}_abs_wells_bw", #build  MODPATH 7 model with the GWF​
                         model_ws=modelws,
                         flowmodel=gwf,
                         exe_name=r"C:\\Users\\SEAM94860\\FLOPY\\general\\mpath7.exe")\

mpbas = fp.modpath.Modpath7Bas(mp,porosity=0.3) # set the porosity for the MODPATH model​

mpsim = fp.modpath.Modpath7Sim(mp, 
                               particlegroups=pg,
                               weaksourceoption="pass_through", # particles that enter a well cell will pass through ​
                               stoptime = 100*365.,# stop time of 100 years
                               trackingdirection='backward',) # backward in time to find where they came from​

mp.write_input() # write the MODPATH input files​
mp.run_model()  

import re

# Fix malformed scientific notation in MODPATH output
mppth_path = os.path.join(modelws, f"{modelname}_abs_wells_bw.mppth")
with open(mppth_path, 'r') as f:
    content = f.read()

# Insert 'E' before negative/positive exponents missing it (e.g., 0.12345-103 -> 0.12345E-103)
fixed = re.sub(r'(\d)([+-]\d{3})', r'\1E\2', content)

with open(mppth_path, 'w') as f:
    f.write(fixed)

#Read And Plot Pathlines​

pmv = fp.plot.PlotMapView(model=gwf)#create a map view plot of the groundwater flow model​
pmv.contour_array(head, levels=range(50,70), colors='C0') # add contours of the head distribution​
fname = os.path.join(modelws, f"{modelname}_abs_wells_bw.mppth") # path to the MODPATH output file ​
p = fp.utils.PathlineFile(fname) # read the pathline file​

for i in range(p.get_maxid() + 1): # loop over all particles​
    pi = p.get_data(partid=i) # get the pathline data for particle i​
    plt.plot(pi['x'], pi['y'], 'C1', lw =1) # plot the pathline of particle i​

plt.title("Pathlines from the Well (Backward)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()