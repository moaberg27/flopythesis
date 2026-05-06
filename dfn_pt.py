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

# Configure logging
import logging
import matplotlib.pyplot as plt

from andfn import ConstantHeadLine

logging.basicConfig(level=logging.INFO)

def make_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

def plot_cdf(data, label, **kwargs):
    sorted_data, cdf = make_cdf(data)
    plt.plot(sorted_data, cdf, label=label, **kwargs)


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
    path = os.path.join(r"C:\Users\SEAM94860\FLOPY\finalflopy\flopythesis\fracs_connected_properties.csv")
    #path = os.path.join(r"C:\Users\seet92866\PycharmProjects\DFN_exjobb\2", "15000fracs.csv")
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
        regbox.rotate(angle=45, axis=[1, 0, 0])
        regbox.rotate(angle=45, axis=[0, 0, 1])
        reg_fracs_in, reg_fracs_out = regbox.check_fractures(dfn.fractures, tree=dfn.tree)

        print(f"Number of fractures in the DFN: {len(dfn.fractures)}")
        dfn.delete_fracture(reg_fracs_out)
        print(f"Number of fractures after deleting those outside the box: {len(dfn.fractures)}")

        regbox.frac_intersections(dfn.fractures, face="left", head=head0)
        regbox.frac_intersections(dfn.fractures, face="right", head=head1)

        dfn.check_connectivity()

        p0=dfn.initiate_plotter()
        dfn.plot_fractures(p0)
        plottable_elements = [e for e in regbox.elements if not isinstance(e, ConstantHeadLine)]
        dfn.plot_elements(p0, elements=plottable_elements)
        regbox.plot(p0)
        p0.show()

        dfn.set_kwargs(COEF_RATIO=0.001, MAX_ITERATIONS=30, MAX_NCOEF=200, MAX_ERROR=5e-4)

        start1 = datetime.datetime.now()
        print("\n---- SOLVE THE DFN ----")
        dfn.solve(unconsolidate=True)

        start2 = datetime.datetime.now()

        print("\n---- GET FLOWS ----")
        sum_flows = regbox.get_total_flow() / 2
        print(f"Total flow through the box: {sum_flows:.2e} m^3/s")

        print("\n---- PLOTTING ----")
        p1 = dfn.initiate_plotter(title=True, off_screen=False, scale=1, axis=True)

        dfn.plot_fractures_head(
            p1, 40, 10, opacity=1, contour=True
        )  # , limits=[200, 400], debug=False)
        regbox.plot(p1)


        # Particle tracking  
        print(f"\n---- PARTICLE TRACKING ----")
        cnt = 0
        ss = []
        sf = []
        vel = []
        el = []
        for e in dfn.elements:
            if isinstance(e, andfn.const_head.ConstantHeadLine):
                if e.q < -1e-16:
                    cnt += 1
                    celement = e
                    z_start = celement.z_array_tracking(10, offset=1e-1)
                    elevation = [0.125, 0.25, 0.5, 0.75, 0.875]
                    ds = 1e-2
                    streamlines, streamlines_fracs, velocities, elements = (
                        dfn.plot_streamline_tracking(
                            p1,
                            z_start,
                            celement.frac0,
                            ds=ds,
                            max_length=5e3,
                            line_width=4,
                            elevation=elevation,
                            remove_false=True,
                            backward=False,
                        )
                    )

                    #if len(streamlines[0]) < 20:
                    #    continue
                    ss.append(streamlines)
                    sf.append(streamlines_fracs)
                    vel.append(velocities)
                    el.append(elements)
                    print(f"Streamline {cnt} starting at fracture {celement.frac0} with head {celement.head:.2f} and flow rate {celement.q:.2e}")
                    if cnt > 100:
                        break
        time = []
        length = []
        for strem, velo, elem in zip(ss, vel, el):
            for v, s, ee in zip(velo, strem, elem):
                if ee is False:
                    continue
                if len(s) == 0:
                    continue
                t, l = dfn.get_travel_time_and_length(s, v)
                time.append(t)
                length.append(l)

        time = np.array(time) / (60*60)  # convert to hours
        length = np.array(length)

        # POROSITY 
        V_bulk = regbox.xl * regbox.yl * regbox.zl
        Q_tot = sum_flows
        print(f"\n---- POROSITY ----")
        print(f"Bulk volume:      {V_bulk:.2e} m^3")
        print(f"Total flow rate:  {Q_tot:.2e} m^3/s")

        # 1: geometrical – all connected fractures inside the box
        phi_geo_all = sum(
            np.pi * f.radius**2 * (f.aperture if f.aperture is not None else 0.0)
            for f in dfn.fractures
        ) / V_bulk
        print(f"Geometrical porosity (all connected):  phi_geo_all  = {phi_geo_all:.4e}")

        # 2: geometrical with flow condition
        # Only fractures that have at least one inflow ConstantHeadLine (q < -1e-16)
        flowing_frac_ids = {
            id(e.frac0)
            for e in dfn.elements
            if isinstance(e, andfn.const_head.ConstantHeadLine) and e.q < -1e-16
        }
        phi_geo_flow = sum(
            np.pi * f.radius**2 * (f.aperture if f.aperture is not None else 0.0)
            for f in dfn.fractures
            if id(f) in flowing_frac_ids
        ) / V_bulk
        print(f"Geometrical porosity (flow-carrying):  phi_geo_flow = {phi_geo_flow:.4e}")

        # 3: effective (kinematic) porosity from mean particle travel time
        tau = np.mean(time) * (60 * 60)  # time is in hours; convert back to seconds
        phi_eff = (tau * Q_tot) / V_bulk
        print(f"Effective (kinematic) porosity:        phi_eff      = {phi_eff:.4e}")


        p1.show()

    end = datetime.datetime.now()
    print(f"\n\nProgram ended at {end}")
    print(f"Time elapsed: {end - start0}")
    print(f"\t-generating: \t{start1 - start0}")
    print(f"\t-solving: \t\t\t{start2 - start1}")
    print(f"\t-plotting: \t\t{end - start2}")

    # Plot break through curve
    time  = time / (24 * 365) # convert to years
    an_len = length[time > 0.1] # only consider travel times greater than 0.1 years to avoid very short paths that may not be representative
    time = time[time > 0.1] 

    plt.figure(figsize=(8, 6))
    plot_cdf(time, label='AnDFN', color='red')
    plt.legend(prop={'size': 14, 'family': 'Times New Roman'})
    plt.xscale('log')
    plt.xlabel('Travel Time [years]', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Cumulative Distribution Function', fontsize=16, fontname='Times New Roman')
    plt.title(
        f"$\\phi_{{geo,all}}$={phi_geo_all:.3e}   "
        f"$\\phi_{{geo,flow}}$={phi_geo_flow:.3e}   "
        f"$\\phi_{{eff}}$={phi_eff:.3e}",
        fontsize=13,
    )
    plt.grid()
    plt.tight_layout()
    plt.tick_params(colors='black', labelsize=14, labelfontfamily="Times New Roman")

    an_len = an_len[an_len > 200] # only consider trace lengths greater than 200 m to focus on more representative paths
    plt.figure(figsize=(8, 6))
    plot_cdf(an_len, label='AnDFN', color='red')
    plt.legend(prop={'size': 14, 'family': 'Times New Roman'})
    plt.xlabel('Trace length [m]', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Cumulative Distribution Function', fontsize=16,
            fontname='Times New Roman')
    plt.title(
        f"$\\phi_{{geo,all}}$={phi_geo_all:.3e}   "
        f"$\\phi_{{geo,flow}}$={phi_geo_flow:.3e}   "
        f"$\\phi_{{eff}}$={phi_eff:.3e}",
        fontsize=13,
    )
    plt.grid()
    plt.tight_layout()
    plt.tick_params(colors='black', labelsize=14, labelfontfamily="Times New Roman")
    plt.show()  
    print("All done!")
