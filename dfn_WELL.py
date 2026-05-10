"""
Notes
-----
Single flow simulation along the x-axis of a rotated region box, with a
ConstantHeadPrism (tunnel) sink (head=0) located at the xy-center of the box,
starting at the top face and descending 100 m along the box z-axis.
Particles are released on the tunnel-fracture intersection elements and tracked
BACKWARD to recover the capture zone. Outputs streamlines, BTC and trace-length
CDFs, plus a top-down view of the pathlines.
"""
 
import datetime
import os
 
import numpy as np
import pandas as pd
 
import andfn
 
import logging
import matplotlib.pyplot as plt
 
from andfn import ConstantHeadLine, ConstantHeadPrism
from andfn import geometry_functions as gf
 
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
 
    # Output folder: DFN_runs/WELL_runs/<timestamp>
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(r"C:\Users\SEAM94860\FLOPY\finalflopy", "DFN_runs", "WELL_runs", run_ts)
    os.makedirs(out_dir, exist_ok=True)
 
    start0 = datetime.datetime.now()
    print("\n---- IMPORT DFN ----")
    print(f"Program started at {start0}")
 
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
 
    # Rotated region-box axes - orthonormalized via Gram-Schmidt
    _x = np.array([0.2021, -0.9796, -0.073], dtype=float)
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
 
    # Tunnel parameters: top of box, descending 100 m along the box z-axis
    box_center = np.array([0.0, 0.0, 0.0])
    zl_box = 500.0
    tunnel_top = box_center + (zl_box / 2.0) * _z
    tunnel_bottom = tunnel_top - 100.0 * _z
    tunnel_radius = 5.0
    tunnel_head = 0.0
    tunnel_n_sides = 8
 
    sim_name = "x"
    sim = {"name": sim_name, "face_in": "left", "face_out": "right"}
 
    print(f"\n========== SIMULATION: flow along {sim_name} (with tunnel) ==========")
    sim_dir = os.path.join(out_dir, f"sim_{sim_name}_well")
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
        zl=zl_box,
    )
 
    reg_fracs_in, reg_fracs_out = regbox.check_fractures(dfn.fractures, tree=dfn.tree)
    print(f"Number of fractures in the DFN: {len(dfn.fractures)}")
    dfn.delete_fracture(reg_fracs_out)
    print(f"Number of fractures after deleting those outside the box: {len(dfn.fractures)}")
 
    regbox.frac_intersections(dfn.fractures, face=sim["face_in"], head=head0)
    regbox.frac_intersections(dfn.fractures, face=sim["face_out"], head=head1)
 
    dfn.check_connectivity()
 
    # ---------------- TUNNEL ----------------
    print("\n---- ADDING TUNNEL (ConstantHeadPrism) ----")
    print(f"Tunnel start (top):    {tunnel_top}")
    print(f"Tunnel end  (bottom):  {tunnel_bottom}")
    print(f"Tunnel length: {np.linalg.norm(tunnel_top - tunnel_bottom):.2f} m")
    tunnel = ConstantHeadPrism(
        label="tunnel",
        radius=tunnel_radius,
        start=tunnel_top,
        end=tunnel_bottom,
        head=tunnel_head,
        n_sides=tunnel_n_sides,
    )
    dfn.add_structure(tunnel)
    print(f"Tunnel intersects {len(tunnel.fracs)} fractures, "
          f"creating {len(tunnel.elements)} CHL elements")
 
    # Geometry preview
    p0 = dfn.initiate_plotter(off_screen=True)
    dfn.plot_fractures(p0)
    plottable_elements = [e for e in regbox.elements if not isinstance(e, ConstantHeadLine)]
    dfn.plot_elements(p0, elements=plottable_elements)
    regbox.plot(p0)
    tunnel.plot(p0)
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
    tunnel_q = sum(e.q for e in tunnel.elements)
    print(f"Tunnel total discharge sum(q): {tunnel_q:.3e} m^3/s")
 
    print("\n---- PLOTTING HEAD ----")
    p1 = dfn.initiate_plotter(title=True, off_screen=True, scale=1, axis=True)
    dfn.plot_fractures_head(p1, 40, 10, opacity=1, contour=True)
    regbox.plot(p1)
    tunnel.plot(p1)
 
    # Particle tracking - BACKWARD from tunnel-fracture CHL elements (sinks)
    print(f"\n---- PARTICLE TRACKING (backward from tunnel) ----")
    cnt = 0
    ss = []
    sf = []
    vel = []
    el = []
    elevation = [0.125, 0.25, 0.5, 0.75, 0.875]
    ds = 1e-2
    for e in tunnel.elements:
        if not isinstance(e, ConstantHeadLine):
            continue
        # tunnel acts as a sink: water leaves the fracture into the tunnel (q < 0)
        if e.q < -1e-16:
            cnt += 1
            z_start = e.z_array_tracking(5, offset=1e-1)
            streamlines, streamlines_fracs, velocities, elements = (
                dfn.plot_streamline_tracking(
                    p1,
                    z_start,
                    e.frac0,
                    ds=ds,
                    max_length=5e3,
                    line_width=4,
                    elevation=elevation,
                    remove_false=True,
                    backward=True,
                    color="red",
                )
            )
            ss.append(streamlines)
            sf.append(streamlines_fracs)
            vel.append(velocities)
            el.append(elements)
            print(
                f"\nTunnel CHL {cnt} on fracture {e.frac0} q={e.q:.2e}"
            )
            if cnt > 100:
                break
    print(f"\nReleased backward streamlines from {cnt} tunnel CHL segments")
 
    # 3D streamlines screenshot
    p1.screenshot(os.path.join(sim_dir, f"streamlines_{sim_name}.png"))
    # Top-down view
    try:
        p1.view_xy()
        p1.screenshot(os.path.join(sim_dir, f"streamlines_{sim_name}_top.png"))
    except Exception as exc:
        print(f"Top-down view failed: {exc}")
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
        tau_median = np.median(time) * 3600.0
        tau_fast = np.min(time) * 3600.0
        tau_slow = np.max(time) * 3600.0
    else:
        tau_mean = tau_median = tau_fast = tau_slow = np.nan
 
    phi_eff_mean = (tau_mean * Q_tot) / V_bulk if Q_tot != 0 else np.nan
    phi_eff_median = (tau_median * Q_tot) / V_bulk if Q_tot != 0 else np.nan
    phi_eff_fast = (tau_fast * Q_tot) / V_bulk if Q_tot != 0 else np.nan
    phi_eff_slow = (tau_slow * Q_tot) / V_bulk if Q_tot != 0 else np.nan
    print(f"Effective (kinematic) porosity (mean): phi_eff_mean      = {phi_eff_mean:.4e}")
    print(f"Effective (kinematic) porosity (median): phi_eff_median = {phi_eff_median:.4e}")
    print(f"Effective (kinematic) porosity (fast): phi_eff_fast = {phi_eff_fast:.4e}")
    print(f"Effective (kinematic) porosity (slow): phi_eff_slow = {phi_eff_slow:.4e}")
 
    _yr = 3600 * 24 * 365
    tau_geo_all_yr = (phi_geo_all * V_bulk / Q_tot) / _yr if Q_tot != 0 else np.nan
    tau_geo_flow_yr = (phi_geo_flow * V_bulk / Q_tot) / _yr if Q_tot != 0 else np.nan
    tau_mean_yr = tau_mean / _yr
    tau_median_yr = tau_median / _yr
    tau_fast_yr = tau_fast / _yr
    tau_slow_yr = tau_slow / _yr
 
    df = pd.DataFrame({"travel_time_hours": time, "trace_length_m": length})
    df.to_csv(os.path.join(sim_dir, f"particles_{sim_name}.csv"), index=False)
 
    summary = {
        "simulation": sim_name,
        "tunnel_q_m3_s": float(tunnel_q),
        "tunnel_n_intersected_fracs": int(len(tunnel.fracs)),
        "tunnel_n_chl_elements": int(len(tunnel.elements)),
        "V_bulk_m3": V_bulk,
        "Q_tot_m3_s": Q_tot,
        "phi_geo_all": phi_geo_all,
        "phi_geo_flow": phi_geo_flow,
        "phi_eff_mean": phi_eff_mean,
        "phi_eff_median": phi_eff_median,
        "phi_eff_fast": phi_eff_fast,
        "phi_eff_slow": phi_eff_slow,
        "tau_geo_all_yr": tau_geo_all_yr,
        "tau_geo_flow_yr": tau_geo_flow_yr,
        "tau_mean_yr": tau_mean_yr,
        "tau_median_yr": tau_median_yr,
        "tau_fast_yr": tau_fast_yr,
        "tau_slow_yr": tau_slow_yr,
        "n_particles": int(len(time)),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(sim_dir, f"summary_{sim_name}.csv"), index=False
    )
 
    time_yr = time / (24 * 365)  # hours -> years
    mask = length > regbox.xl
    an_len = length[mask]
    time_yr_filt = time_yr[mask]
 
    # BTC plot
    plt.figure(figsize=(8, 6))
    if len(time_yr_filt) > 0:
        plot_cdf(time_yr_filt, label="AnDFN backward from tunnel", color="red")
    plt.legend(prop={"size": 14, "family": "Times New Roman"})
    plt.xscale("log")
    plt.xlabel("Travel Time [years]", fontsize=16, fontname="Times New Roman")
    plt.ylabel("Cumulative Distribution Function", fontsize=16, fontname="Times New Roman")
    plt.title(
        f"[{sim_name} + tunnel]  "
        f"$\\tau_{{geo,all}}$={tau_geo_all_yr:.2e} yr   "
        f"$\\tau_{{mean}}$={tau_mean_yr:.2e} yr   "
        f"$\\tau_{{median}}$={tau_median_yr:.2e} yr   "
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
        plot_cdf(an_len, label="AnDFN backward from tunnel", color="red")
    plt.legend(prop={"size": 14, "family": "Times New Roman"})
    plt.xlabel("Trace length [m]", fontsize=16, fontname="Times New Roman")
    plt.ylabel("Cumulative Distribution Function", fontsize=16, fontname="Times New Roman")
    plt.title(f"Trace lengths - flow along {sim_name} (tunnel capture zone)", fontsize=13)
    plt.grid()
    plt.tight_layout()
    plt.tick_params(colors="black", labelsize=14, labelfontfamily="Times New Roman")
    plt.savefig(os.path.join(sim_dir, f"tracelength_{sim_name}.png"), dpi=150)
    plt.close()
 
    # Top-down matplotlib pathline plot (xy plane)
    print("\n---- TOP-DOWN PATHLINE PLOT ----")
    plt.figure(figsize=(8, 8))
    valid = 0
    for strem_list, frac_list, elem_list in zip(ss, sf, el):
        for s_per_seed, f_per_seed, ee in zip(strem_list, frac_list, elem_list):
            if ee is False or len(s_per_seed) == 0:
                continue
            for s_arr, f in zip(s_per_seed, f_per_seed):
                arr = np.asarray(s_arr)
                if arr.size == 0:
                    continue
                pts3d = gf.map_2d_to_3d(arr, f)
                plt.plot(pts3d[:, 0], pts3d[:, 1], color="red", lw=0.6, alpha=0.7)
            valid += 1
    plt.scatter([tunnel_top[0], tunnel_bottom[0]],
                [tunnel_top[1], tunnel_bottom[1]],
                c="black", marker="o", s=40, zorder=5,
                label=f"tunnel axis (r={tunnel_radius} m, L=100 m)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x [m]", fontsize=14, fontname="Times New Roman")
    plt.ylabel("y [m]", fontsize=14, fontname="Times New Roman")
    plt.title(f"Pathlines (top view) - {valid} backward streamlines", fontsize=13)
    plt.legend(prop={"size": 12, "family": "Times New Roman"})
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(sim_dir, f"pathlines_top_{sim_name}.png"), dpi=150)
    plt.close()
 
    end_sim = datetime.datetime.now()
    print(f"\nSimulation {sim_name} done.")
    print(f"\t-generating: \t{start1 - start0}")
    print(f"\t-solving: \t{start2 - start1}")
    print(f"\t-tracking+plot: {end_sim - start2}")
    print(f"All outputs saved in: {sim_dir}")
    print("All done!")
 
 