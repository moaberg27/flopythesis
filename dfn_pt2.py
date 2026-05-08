"""
Notes
-----
This is an example of a model. CLAUDE CODE EDITED
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
from andfn.geometry_functions import map_z_line_to_chi
 
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
 
    # ---- particle-tracking / porosity options ----
    # Point 1: flow-weighted mean residence time instead of unweighted np.mean(time)
    USE_FLOW_WEIGHTED_TAU = True
    # Point 2: keep tau and Q_tot consistent
    #   "A" -> remove the inflow-line cap (track them all)
    #   "B" -> keep the cap, recompute Q_tot from the lines actually tracked
    INFLOW_LINE_MODE = "B"
    INFLOW_LINE_CAP = 100  # only used if INFLOW_LINE_MODE == "B"
    # Point 3: include particles that hit max_length / got stuck (right-censored)
    INCLUDE_FAILED_STREAMLINES = False
    # Point 4: only count streamlines that actually reached the outflow face
    FILTER_OUTFLOW_ONLY = True
    N_RELEASE_PER_LINE = 10
    RELEASE_OFFSET = 1e-1
 
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
        tracked_lines = []          # one ConstantHeadLine per ss[i]
        release_weights = []        # one np.ndarray of per-particle weights per ss[i]
        for e in dfn.elements:
            if isinstance(e, andfn.const_head.ConstantHeadLine):
                if e.q < -1e-16:
                    cnt += 1
                    celement = e
 
                    z_start = celement.z_array_tracking(
                        N_RELEASE_PER_LINE, offset=RELEASE_OFFSET
                    )
 
                    # Per-particle release weight for flow-weighted tau (Point 1).
                    # Release is chi-uniform (θ-uniform on |chi|=1+offset), so the
                    # arclength element each particle represents is
                    #   dℓ = |dz/dθ|·dθ = ¼·L·|chi − 1/chi|·(2π/n)
                    # Weight ∝ |w_i| · dℓ.
                    L_line = np.abs(celement.endpoints0[1] - celement.endpoints0[0])
                    chi_at_start = map_z_line_to_chi(z_start, celement.endpoints0)
                    dl_dtheta = 0.25 * L_line * np.abs(chi_at_start - 1.0 / chi_at_start)
                    dtheta = 2.0 * np.pi / N_RELEASE_PER_LINE
                    w_at_start = np.array(
                        [np.abs(celement.frac0.calc_w(z)) for z in z_start]
                    )
                    weights_per_release = w_at_start * dl_dtheta * dtheta
 
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
 
                    # plot_streamline_tracking returns one entry per
                    # (start_point, elevation) pair, in start-major order.
                    # Replicate the per-release weight across elevations to match.
                    n_elev = len(elevation)
                    weights_full = np.repeat(weights_per_release, n_elev)
 
                    ss.append(streamlines)
                    sf.append(streamlines_fracs)
                    vel.append(velocities)
                    el.append(elements)
                    tracked_lines.append(celement)
                    release_weights.append(weights_full)
                    print(f"Streamline {cnt} starting at fracture {celement.frac0} with head {celement.head:.2f} and flow rate {celement.q:.2e}")
 
                    # Point 2: cap behaviour
                    if INFLOW_LINE_MODE == "B" and cnt >= INFLOW_LINE_CAP:
                        break
                    # mode "A" -> no cap
 
        # Aggregate per-streamline travel time, length, and weight
        time = []
        length = []
        weight = []
        traversed_frac_ids = set()  # fractures actually visited by a kept streamline
        n_total = 0
        n_failed = 0
        n_wrong_outlet = 0
        for strem, velo, elem, fracs_line, w_line, line in zip(
            ss, vel, el, sf, release_weights, tracked_lines
        ):
            for v, s, ee, frac_chain, w_i in zip(
                velo, strem, elem, fracs_line, w_line
            ):
                n_total += 1
                failed = ee is False or len(s) == 0
                if failed:
                    n_failed += 1
                    if not INCLUDE_FAILED_STREAMLINES:
                        continue
                # Point 4: only count streamlines that reached the outflow face
                if FILTER_OUTFLOW_ONLY and not failed:
                    if not (isinstance(ee, ConstantHeadLine) and ee.head == head1):
                        n_wrong_outlet += 1
                        continue
                if len(s) == 0:
                    continue
                t, l = dfn.get_travel_time_and_length(s, v)
                time.append(t)
                length.append(l)
                weight.append(w_i)
                # Record every fracture this streamline traversed
                for fr in frac_chain:
                    traversed_frac_ids.add(id(fr))
 
        print(f"Streamlines: total={n_total}, failed/stuck={n_failed}, wrong-outlet dropped={n_wrong_outlet}, kept={len(time)}")
 
        time = np.array(time) / (60 * 60)  # to hours
        length = np.array(length)
        weight = np.array(weight)
 
        # POROSITY
        V_bulk = regbox.xl * regbox.yl * regbox.zl
        # Point 2: keep Q_tot consistent with the streamline ensemble
        if INFLOW_LINE_MODE == "B":
            Q_tot = float(np.sum([np.abs(line.q) for line in tracked_lines]))
        else:
            Q_tot = sum_flows
        print(f"\n---- POROSITY ----")
        print(f"Bulk volume:      {V_bulk:.2e} m^3")
        print(f"Total flow rate:  {Q_tot:.2e} m^3/s  (mode {INFLOW_LINE_MODE})")
 
        # Type 1: geometrical – all connected fractures inside the box
        phi_geo_all = sum(
            np.pi * f.radius**2 * (f.aperture if f.aperture is not None else 0.0)
            for f in dfn.fractures
        ) / V_bulk
        print(f"Geometrical porosity (all connected):  phi_geo_all  = {phi_geo_all:.4e}")
 
        # Type 2: geometrical with flow condition – fractures actually visited
        # by a kept streamline (the flow-carrying backbone seen by the tracer).
        phi_geo_flow = sum(
            np.pi * f.radius**2 * (f.aperture if f.aperture is not None else 0.0)
            for f in dfn.fractures
            if id(f) in traversed_frac_ids
        ) / V_bulk
        print(f"Flow-carrying fractures (streamline-traversed): {len(traversed_frac_ids)} / {len(dfn.fractures)}")
        print(f"Geometrical porosity (flow-carrying):  phi_geo_flow = {phi_geo_flow:.4e}")
 
        # Type 3: effective (kinematic) porosity from mean particle travel time
        time_s = time * (60 * 60)  # back to seconds
        if USE_FLOW_WEIGHTED_TAU and weight.sum() > 0:
            tau = float(np.sum(time_s * weight) / np.sum(weight))
            tau_label = "flow-weighted"
        else:
            tau = float(np.mean(time_s))
            tau_label = "unweighted"
        phi_eff = (tau * Q_tot) / V_bulk
        print(f"Mean residence time ({tau_label}): tau = {tau:.3e} s")
        print(f"Effective (kinematic) porosity:        phi_eff      = {phi_eff:.4e}")
 
 
        p1.show()
 
    end = datetime.datetime.now()
    print(f"\n\nProgram ended at {end}")
    print(f"Time elapsed: {end - start0}")
    print(f"\t-generating: \t{start1 - start0}")
    print(f"\t-solving: \t\t\t{start2 - start1}")
    print(f"\t-plotting: \t\t{end - start2}")
 
    # Plot break through curve
    time  = time / (24 * 365)
    an_len = length[time > 0.1]
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
 
    an_len = an_len[an_len > 200]
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
 
 
 