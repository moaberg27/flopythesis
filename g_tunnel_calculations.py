"""Calculate tunnel inflow from the continuum model and tunnel geometry."""

from __future__ import annotations

import math
from pathlib import Path

import flopy
import numpy as np
import pandas as pd
import pyvista as pv
import geopandas as gpd
from flopy.export.vtk import Vtk
from shapely.geometry import Polygon

from tunnel import (
    TUNNEL_CENTER_DEPTH_M,
    TUNNEL_RADIUS_M,
    build_tunnel_centerline,
    write_tunnel_shapefile,
)


ROOT = Path(__file__).resolve().parent
SIM_WS = ROOT / "continuum_ws"
MODEL_NAME = "cube"
OUTPUT_DIR = ROOT / "tunneldata"
OUTPUT_CSV = OUTPUT_DIR / "tunnel_inflow.csv"
IMAGES_DIR = ROOT / "images"


def load_continuum_model(sim_ws: Path, model_name: str) -> tuple[flopy.mf6.MFSimulation, flopy.mf6.ModflowGwf]:
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws), verbosity_level=0)
    gwf = sim.get_model(model_name)
    if gwf is None:
        raise ValueError(f"Model '{model_name}' was not found in {sim_ws}")
    return sim, gwf


def load_head_array(sim_ws: Path, model_name: str) -> np.ndarray:
    head_path = sim_ws / f"{model_name}.hds"
    if not head_path.exists():
        raise FileNotFoundError(f"Head file not found: {head_path}")

    head_file = flopy.utils.HeadFile(str(head_path))
    return head_file.get_data(kstpkper=(0, 0))


def model_layer_bounds(gwf: flopy.mf6.ModflowGwf) -> tuple[float, np.ndarray]:
    top = np.asarray(gwf.dis.top.array, dtype=float)
    botm = np.asarray(gwf.dis.botm.array, dtype=float)

    if top.ndim > 0:
        top_value = float(np.nanmean(top))
    else:
        top_value = float(top)

    layer_bottoms = np.array([float(np.nanmean(botm[layer_index]))
                             for layer_index in range(botm.shape[0])], dtype=float)
    return top_value, layer_bottoms


def layer_overlap_fractions(top: float, botm: np.ndarray, tunnel_center_depth_m: float, tunnel_radius_m: float) -> np.ndarray:
    tunnel_center_elevation = -float(tunnel_center_depth_m)
    tunnel_top = tunnel_center_elevation + tunnel_radius_m
    tunnel_bottom = tunnel_center_elevation - tunnel_radius_m

    fractions = np.zeros(len(botm), dtype=float)
    for layer_index in range(len(botm)):
        layer_top = float(top) if layer_index == 0 else float(
            botm[layer_index - 1])
        layer_bottom = float(botm[layer_index])
        overlap = max(0.0, min(layer_top, tunnel_top) -
                      max(layer_bottom, tunnel_bottom))
        fractions[layer_index] = overlap / (2.0 * tunnel_radius_m)

    return fractions


def extract_disv_head_value(head: np.ndarray, layer_index: int, cell_index: int) -> float:
    return float(head[layer_index, 0, cell_index])


def compute_inflow_rows(gwf: flopy.mf6.ModflowGwf, head: np.ndarray) -> pd.DataFrame:
    modelgrid = gwf.modelgrid
    if getattr(modelgrid, "grid_type", "") != "vertex":
        raise ValueError(
            "This script currently expects the DISV vertex grid from continuum_ws.")

    npf = gwf.get_package("npf")
    k_base = np.asarray(npf.k.array, dtype=float)
    k22 = np.asarray(npf.k22.array, dtype=float)
    k33 = np.asarray(npf.k33.array, dtype=float)

    k_horizontal = float(np.nanmean(k_base * k22))
    k_vertical = float(np.nanmean(k_base * k33))
    k_radial = math.sqrt(k_horizontal * k_vertical)

    if TUNNEL_RADIUS_M <= 0.0:
        raise ValueError("Tunnel radius must be positive.")

    influence_radius_m = max(5.0 * TUNNEL_RADIUS_M, 20.0)
    denominator = math.log(influence_radius_m / TUNNEL_RADIUS_M)
    if denominator <= 0.0:
        raise ValueError("Invalid Goodman influence radius.")

    tunnel_line = build_tunnel_centerline()
    tunnel_elevation = -float(TUNNEL_CENTER_DEPTH_M)

    top, botm = model_layer_bounds(gwf)
    layer_fractions = layer_overlap_fractions(
        top, botm, TUNNEL_CENTER_DEPTH_M, TUNNEL_RADIUS_M)

    rows: list[dict[str, float | int]] = []
    ncpl = int(getattr(modelgrid, "ncpl"))

    for layer_index, layer_fraction in enumerate(layer_fractions):
        if layer_fraction <= 0.0:
            continue

        for cell_index in range(ncpl):
            cell_polygon = Polygon(modelgrid.get_cell_vertices(cell_index))
            intersection = tunnel_line.intersection(cell_polygon)
            segment_length_m = float(intersection.length)
            if segment_length_m <= 0.0:
                continue

            cell_head_m = extract_disv_head_value(
                head, layer_index, cell_index)
            delta_head_m = max(cell_head_m - tunnel_elevation, 0.0)
            inflow_per_meter_m3_s = (
                2.0 * math.pi * k_radial * delta_head_m) / denominator
            inflow_m3_s = inflow_per_meter_m3_s * segment_length_m * layer_fraction

            centroid = intersection.centroid
            rows.append(
                {
                    "layer": layer_index,
                    "cell_index": cell_index,
                    "segment_length_m": segment_length_m,
                    "segment_center_x_m": float(centroid.x),
                    "segment_center_y_m": float(centroid.y),
                    "layer_fraction": float(layer_fraction),
                    "cell_head_m": cell_head_m,
                    "tunnel_head_m": tunnel_elevation,
                    "delta_head_m": delta_head_m,
                    "k_horizontal_m_s": k_horizontal,
                    "k_vertical_m_s": k_vertical,
                    "k_radial_m_s": k_radial,
                    "inflow_m3_s": inflow_m3_s,
                    "inflow_m3_d": inflow_m3_s * 86400.0,
                }
            )

    if not rows:
        raise ValueError("No tunnel segments intersected the model grid.")

    df = pd.DataFrame(rows).sort_values(
        ["layer", "cell_index"]).reset_index(drop=True)
    df["cumulative_inflow_m3_s"] = df["inflow_m3_s"].cumsum()
    df["cumulative_inflow_m3_d"] = df["inflow_m3_d"].cumsum()
    return df


def build_tunnel_mesh_from_shapefile(tunnel_path: Path, top_elevation: float = 0.0) -> pv.PolyData | None:
    if not tunnel_path.exists():
        return None

    gdf = gpd.read_file(tunnel_path)
    if gdf.empty:
        return None

    tunnel_geom = gdf.geometry.iloc[0]
    tunnel_depth_m = float(gdf["depth_m"].iloc[0])
    tunnel_radius_m = float(gdf["radius_m"].iloc[0])

    if not isinstance(tunnel_geom, Polygon):
        return None

    coords = np.asarray(tunnel_geom.exterior.coords[:-1], dtype=float)
    z_center = top_elevation - tunnel_depth_m
    z_top = z_center + tunnel_radius_m
    z_bottom = z_center - tunnel_radius_m

    bottom_points = np.column_stack(
        [coords[:, 0], coords[:, 1], np.full(len(coords), z_bottom)]
    )
    top_points = np.column_stack(
        [coords[:, 0], coords[:, 1], np.full(len(coords), z_top)]
    )
    all_points = np.vstack([bottom_points, top_points])
    n_side = len(coords)

    faces: list[list[int]] = []
    for i in range(n_side):
        i_next = (i + 1) % n_side
        faces.append([4, i, i_next, i_next + n_side, i + n_side])

    faces.append([n_side] + list(range(n_side)))
    faces.append([n_side] + list(range(n_side, 2 * n_side))[::-1])

    faces_array = np.concatenate(
        [np.asarray(face, dtype=int) for face in faces])
    return pv.PolyData(all_points, faces_array)


def plot_continuum_model_with_tunnel(gwf: flopy.mf6.ModflowGwf) -> None:
    vtk = Vtk(model=gwf, binary=False, smooth=False)
    vtk.add_model(gwf)
    grid = vtk.to_pyvista()

    head_path = SIM_WS / f"{MODEL_NAME}.hds"
    if head_path.exists():
        hds = flopy.utils.HeadFile(str(head_path))
        head = hds.get_data(kstpkper=(0, 0)).flatten()
        grid["head"] = head
        scalar_name = "head"
        cmap = "viridis"
    else:
        ncpl = int(getattr(gwf.modelgrid, "ncpl", 0))
        nlay = int(getattr(gwf.modelgrid, "nlay", 1))
        grid["layer_id"] = np.repeat(np.arange(nlay), ncpl)
        scalar_name = "layer_id"
        cmap = "plasma"

    tunnel_mesh = build_tunnel_mesh_from_shapefile(
        OUTPUT_DIR / "tunnel.shp", top_elevation=0.0)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = IMAGES_DIR / "continuum_model_with_tunnel.png"

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, scalars=scalar_name, cmap=cmap,
                     show_edges=True, opacity=1.0)
    if tunnel_mesh is not None:
        plotter.add_mesh(tunnel_mesh, color="firebrick",
                         opacity=0.65, show_edges=False)
    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = "iso"
    plotter.show(screenshot=str(output_path))
    plotter.close()
    print(f"Saved continuum model plot with tunnel to: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (OUTPUT_DIR / "tunnel.shp").exists():
        write_tunnel_shapefile(OUTPUT_DIR)

    _, gwf = load_continuum_model(SIM_WS, MODEL_NAME)
    head = load_head_array(SIM_WS, MODEL_NAME)

    inflow_df = compute_inflow_rows(gwf, head)
    inflow_df.to_csv(OUTPUT_CSV, index=False)

    total_inflow_m3_s = float(inflow_df["inflow_m3_s"].sum())
    total_inflow_m3_d = float(inflow_df["inflow_m3_d"].sum())

    print(f"Tunnel centerline: {build_tunnel_centerline().wkt}")
    print(f"Tunnel radius: {TUNNEL_RADIUS_M:.2f} m")
    print(f"Tunnel center depth: {TUNNEL_CENTER_DEPTH_M:.2f} m")
    print(
        f"Total inflow: {total_inflow_m3_s:.6e} m3/s ({total_inflow_m3_d:.6e} m3/d)")
    print(f"Saved detailed inflow table to: {OUTPUT_CSV}")

    plot_continuum_model_with_tunnel(gwf)


if __name__ == "__main__":
    main()
