"""
Notes
-----
This is an example of a model.
Runs 3 flow simulations (boundary along x, y, z) with a rotated region box,
saves trace lengths, travel times and plots for each simulation in pt_runs/,
and produces final comparison BTC and trace-length CDF plots.
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
    save = True
    scale = 1
    tracking = True
    animate = False
 
    ncoef = 10 * 0 + 10
    nint = ncoef * 2
 
    # Output folder
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(r"C:\Users\SEAM94860\FLOPY\finalflopy", "DFN_runs", "PT_runs", run_ts)
    os.makedirs(out_dir, exist_ok=True)
 
    start0 = datetime.datetime.now()
    print("\n---- IMPORT DFN ----")
    print(f"Program started at {start0}")
 
    # load the geometry once
    dfn_org = andfn.DFN("DFN test FracMan", discharge_int=50)
    path = os.path.join(r"C:\Users\SEAM94860\FLOPY\finalflopy\flopythesis\dfn_4.csv")
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
        remove_isolated=False,
    )
 
    # Rotated region-box axes (provided) - orthonormalized via Gram-Schmidt
    # to fix small non-orthogonalities in the input values.
    _x = np.array([0.2021, -0.9766, -0.073], dtype=float)
    _y = np.array([-0.9793, -0.2025, -0.002], dtype=float)
    _x /= np.linalg.norm(_x)
    _y -= np.dot(_y, _x) * _x
    _y /= np.linalg.norm(_y)
    _z = np.cross(_x, _y)
    _z /= np.linalg.norm(_z)
    x_vec = _x.tolist()
    y_vec = _y.tolist()
    z_vec = _z.tolist()
 
    head0 = 100
    head1 = 200
 
    # Three simulations: boundary faces along each axis of the rotated box
    # face pairs: ("left","right") -> x-axis;  ("front","back") -> y-axis;  ("bottom","top") -> z-axis
    simulations = [
        {"name": "x", "face_in": "left",   "face_out": "right"},
        {"name": "y", "face_in": "front",  "face_out": "back"},
        {"name": "z", "face_in": "bottom", "face_out": "top"},
    ]
 
    # Per-simulation collected travel times / lengths for the final comparison
    results = {}
 
    for sim in simulations:
        sim_name = sim["name"]
        print(f"\n========== SIMULATION: flow along {sim_name} ==========")
 
        sim_dir = os.path.join(out_dir, f"sim_{sim_name}")
        os.makedirs(sim_dir, exist_ok=True)
 
        dfn = andfn.DFN(f"Copy_{sim_name}", discharge_int=50)
        dfn.add_fracture(dfn_org.fractures)
 
        print("Adding constant head boundary conditions")
        regbox = andfn.RectangularRegion(
            label="box",
            center=[0, 0, 0],
            x_vec=x_vec,
            y_vec=y_vec,
            z_vec=z_vec,
            xl=500,
            yl=500,
            zl=500,
        )
 
        reg_fracs_in, reg_fracs_out = regbox.check_fractures(dfn.fractures, tree=dfn.tree)
        print(f"Number of fractures in the DFN: {len(dfn.fractures)}")
        dfn.delete_fracture(reg_fracs_out)
        print(f"Number of fractures after deleting those outside the box: {len(dfn.fractures)}")
 
        regbox.frac_intersections(dfn.fractures, face=sim["face_in"], head=head0)
        regbox.frac_intersections(dfn.fractures, face=sim["face_out"], head=head1)
 
        dfn.check_connectivity()
 
        # Geometry preview
        p0 = dfn.initiate_plotter(off_screen=True)
        dfn.plot_fractures(p0)
        plottable_elements = [e for e in regbox.elements if not isinstance(e, ConstantHeadLine)]
        dfn.plot_elements(p0, elements=plottable_elements)
        regbox.plot(p0)
        p0.screenshot(os.path.join(sim_dir, f"geometry_{sim_name}.png"))
        try:
            p0.close()
        except Exception:
            pass
 
        dfn.set_kwargs(COEF_RATIO=0.001, MAX_ITERATIONS=30, MAX_NCOEF=200, MAX_ERROR=5e-4)
 
        start1 = datetime.datetime.now()
        print("\n---- SOLVE THE DFN ----")
        dfn.solve(unconsolidate=True)
 
        start2 = datetime.datetime.now()
 
        print("\n---- GET FLOWS ----")
        sum_flows = regbox.get_total_flow() / 2
        print(f"Total flow through the box: {sum_flows:.2e} m^3/s")
 
        print("\n---- PLOTTING HEAD ----")
        p1 = dfn.initiate_plotter(title=True, off_screen=True, scale=1, axis=True)
        dfn.plot_fractures_head(p1, 40, 10, opacity=1, contour=True)
        regbox.plot(p1)
 
        # Particle tracking
        print(f"\n---- PARTICLE TRACKING ({sim_name}) ----")
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
                    z_start = celement.z_array_tracking(5, offset=1e-1)
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
                    ss.append(streamlines)
                    sf.append(streamlines_fracs)
                    vel.append(velocities)
                    el.append(elements)
                    print(
                        f"Streamline {cnt} starting at fracture {celement.frac0} "
                        f"with head {celement.head:.2f} and flow rate {celement.q:.2e}"
                    )
                    if cnt > 100:
                        break
 
        # Save streamlines plot
        p1.screenshot(os.path.join(sim_dir, f"streamlines_{sim_name}.png"))
        try:
            p1.close()
        except Exception:
            pass
 
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
 
        time = np.array(time) / (60 * 60)  # convert to hours
        length = np.array(length)
 
        # POROSITY
        V_bulk = regbox.xl * regbox.yl * regbox.zl
        Q_tot = sum_flows
        print(f"\n---- POROSITY ({sim_name}) ----")
        print(f"Bulk volume:      {V_bulk:.2e} m^3")
        print(f"Total flow rate:  {Q_tot:.2e} m^3/s")
 
        phi_geo_all = sum(
            np.pi * f.radius**2 * (f.aperture if f.aperture is not None else 0.0)
            for f in dfn.fractures
        ) / V_bulk
        print(f"Geometrical porosity (all connected):  phi_geo_all  = {phi_geo_all:.4e}")
 
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
 
        if len(time) > 0:
            tau_mean = np.mean(time) * 3600.0
            tau_fast = np.min(time) * 3600.0
            tau_slow = np.max(time) * 3600.0
        else:
            tau_mean = tau_fast = tau_slow = np.nan
 
        phi_eff = (tau_mean * Q_tot) / V_bulk if Q_tot != 0 else np.nan
        phi_eff_fast = (tau_fast * Q_tot) / V_bulk if Q_tot != 0 else np.nan
        phi_eff_slow = (tau_slow * Q_tot) / V_bulk if Q_tot != 0 else np.nan
        print(f"Effective (kinematic) porosity (mean): phi_eff      = {phi_eff:.4e}")
        print(f"Effective (kinematic) porosity (fast): phi_eff_fast = {phi_eff_fast:.4e}")
        print(f"Effective (kinematic) porosity (slow): phi_eff_slow = {phi_eff_slow:.4e}")
 
        _yr = 3600 * 24 * 365
        tau_geo_all_yr = (phi_geo_all * V_bulk / Q_tot) / _yr if Q_tot != 0 else np.nan
        tau_geo_flow_yr = (phi_geo_flow * V_bulk / Q_tot) / _yr if Q_tot != 0 else np.nan
        tau_mean_yr = tau_mean / _yr
        tau_fast_yr = tau_fast / _yr
        tau_slow_yr = tau_slow / _yr
 
        # Save particle data (travel times in hours, lengths in m)
        df = pd.DataFrame({"travel_time_hours": time, "trace_length_m": length})
        df.to_csv(os.path.join(sim_dir, f"particles_{sim_name}.csv"), index=False)
 
        # Save porosities / summary
        summary = {
            "simulation": sim_name,
            "V_bulk_m3": V_bulk,
            "Q_tot_m3_s": Q_tot,
            "phi_geo_all": phi_geo_all,
            "phi_geo_flow": phi_geo_flow,
            "phi_eff_mean": phi_eff,
            "phi_eff_fast": phi_eff_fast,
            "phi_eff_slow": phi_eff_slow,
            "tau_geo_all_yr": tau_geo_all_yr,
            "tau_geo_flow_yr": tau_geo_flow_yr,
            "tau_mean_yr": tau_mean_yr,
            "tau_fast_yr": tau_fast_yr,
            "tau_slow_yr": tau_slow_yr,
            "n_particles": int(len(time)),
        }
        pd.DataFrame([summary]).to_csv(
            os.path.join(sim_dir, f"summary_{sim_name}.csv"), index=False
        )
 
        # Per-simulation BTC and trace-length plots
        time_yr = time / (24 * 365)  # hours -> years
        mask = length > regbox.xl
        an_len = length[mask]
        time_yr_filt = time_yr[mask]
 
        # BTC plot
        plt.figure(figsize=(8, 6))
        if len(time_yr_filt) > 0:
            plot_cdf(time_yr_filt, label=f"AnDFN ({sim_name})", color="red")
        plt.legend(prop={"size": 14, "family": "Times New Roman"})
        plt.xscale("log")
        plt.xlabel("Travel Time [years]", fontsize=16, fontname="Times New Roman")
        plt.ylabel("Cumulative Distribution Function", fontsize=16, fontname="Times New Roman")
        plt.title(
            f"[{sim_name}]  "
            f"$\\tau_{{geo,all}}$={tau_geo_all_yr:.2e} yr   "
            f"$\\tau_{{geo,flow}}$={tau_geo_flow_yr:.2e} yr   "
            f"$\\tau_{{mean}}$={tau_mean_yr:.2e} yr   "
            f"$\\tau_{{fast}}$={tau_fast_yr:.2e} yr   "
            f"$\\tau_{{slow}}$={tau_slow_yr:.2e} yr",
            fontsize=11,
        )
        plt.grid()
        plt.tight_layout()
        plt.tick_params(colors="black", labelsize=14, labelfontfamily="Times New Roman")
        plt.savefig(os.path.join(sim_dir, f"btc_{sim_name}.png"), dpi=150)
        plt.close()
 
        # Trace length plot
        plt.figure(figsize=(8, 6))
        if len(an_len) > 0:
            plot_cdf(an_len, label=f"AnDFN ({sim_name})", color="red")
        plt.legend(prop={"size": 14, "family": "Times New Roman"})
        plt.xlabel("Trace length [m]", fontsize=16, fontname="Times New Roman")
        plt.ylabel("Cumulative Distribution Function", fontsize=16, fontname="Times New Roman")
        plt.title(f"Trace lengths - flow along {sim_name}", fontsize=13)
        plt.grid()
        plt.tight_layout()
        plt.tick_params(colors="black", labelsize=14, labelfontfamily="Times New Roman")
        plt.savefig(os.path.join(sim_dir, f"tracelength_{sim_name}.png"), dpi=150)
        plt.close()
 
        results[sim_name] = {
            "time_yr": time_yr_filt,
            "length": an_len,
            "summary": summary,
        }
 
        end_sim = datetime.datetime.now()
        print(f"\nSimulation {sim_name} done.")
        print(f"\t-generating: \t{start1 - start0}")
        print(f"\t-solving: \t{start2 - start1}")
        print(f"\t-tracking+plot: {end_sim - start2}")
 
    # ===================== Final comparison plots =====================
    print("\n---- FINAL COMPARISON PLOTS ----")
    colors = {"x": "red", "y": "green", "z": "blue"}
 
    # Combined BTC
    plt.figure(figsize=(8, 6))
    for name, r in results.items():
        if len(r["time_yr"]) > 0:
            plot_cdf(r["time_yr"], label=f"flow {name}", color=colors[name])
    plt.legend(prop={"size": 14, "family": "Times New Roman"})
    plt.xscale("log")
    plt.xlabel("Travel Time [years]", fontsize=16, fontname="Times New Roman")
    plt.ylabel("Cumulative Distribution Function", fontsize=16, fontname="Times New Roman")
    plt.title("BTC comparison - flow along x, y, z", fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.tick_params(colors="black", labelsize=14, labelfontfamily="Times New Roman")
    plt.savefig(os.path.join(out_dir, "btc_comparison.png"), dpi=150)
    plt.close()
 
    # Combined trace-length CDF
    plt.figure(figsize=(8, 6))
    for name, r in results.items():
        if len(r["length"]) > 0:
            plot_cdf(r["length"], label=f"flow {name}", color=colors[name])
    plt.legend(prop={"size": 14, "family": "Times New Roman"})
    plt.xlabel("Trace length [m]", fontsize=16, fontname="Times New Roman")
    plt.ylabel("Cumulative Distribution Function", fontsize=16, fontname="Times New Roman")
    plt.title("Trace length comparison - flow along x, y, z", fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.tick_params(colors="black", labelsize=14, labelfontfamily="Times New Roman")
    plt.savefig(os.path.join(out_dir, "tracelength_comparison.png"), dpi=150)
    plt.close()
 
    # Combined summary table
    summary_df = pd.DataFrame([r["summary"] for r in results.values()])
    summary_df.to_csv(os.path.join(out_dir, "summary_all.csv"), index=False)
 
    end = datetime.datetime.now()
    print(f"\nProgram ended at {end}")
    print(f"Total elapsed: {end - start0}")
    print(f"All outputs saved in: {out_dir}")
    print("All done!")
 
 