"""Tunnel geometry helpers used by the inflow calculation scripts."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent
TUNNEL_OUTPUT_DIR = ROOT / "tunneldata"
TUNNEL_FILENAME = "tunnel.shp"

TUNNEL_CENTER_DEPTH_M = 80.0
TUNNEL_RADIUS_M = 4.0
TUNNEL_EXTEND_M = 1.0
TUNNEL_SPACING_M = 0.5
X0, Y0 = 0.0, 50.0
X1, Y1 = 100.0, 50.0


def build_tunnel_endpoints() -> tuple[tuple[float, float], tuple[float, float]]:
    dx = X1 - X0
    dy = Y1 - Y0
    length = (dx**2 + dy**2) ** 0.5
    if length == 0.0:
        raise ValueError("Tunnel endpoints must not be identical.")

    ux = dx / length
    uy = dy / length
    start = (X0 - TUNNEL_EXTEND_M * ux, Y0 - TUNNEL_EXTEND_M * uy)
    end = (X1 + TUNNEL_EXTEND_M * ux, Y1 + TUNNEL_EXTEND_M * uy)
    return start, end


def build_tunnel_centerline() -> LineString:
    start, end = build_tunnel_endpoints()
    return LineString([start, end])


def build_tunnel_polygon():
    start, end = build_tunnel_endpoints()
    line = LineString([start, end])
    length = line.length
    num_circles = max(int(length / TUNNEL_SPACING_M) + 1, 2)

    circles = []
    for i in range(num_circles):
        t = i / (num_circles - 1) if num_circles > 1 else 0.0
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        circles.append(Point(x, y).buffer(TUNNEL_RADIUS_M))

    return unary_union(circles)


def build_tunnel_gdf() -> gpd.GeoDataFrame:
    start, end = build_tunnel_endpoints()
    tunnel_poly = build_tunnel_polygon()

    return gpd.GeoDataFrame(
        {
            "id": [1],
            "depth_m": [TUNNEL_CENTER_DEPTH_M],
            "radius_m": [TUNNEL_RADIUS_M],
            "x0": [start[0]],
            "y0": [start[1]],
            "x1": [end[0]],
            "y1": [end[1]],
        },
        geometry=[tunnel_poly],
        crs="EPSG:3006",
    )


def write_tunnel_shapefile(output_dir: Path | str = TUNNEL_OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shapefile_path = output_dir / TUNNEL_FILENAME
    build_tunnel_gdf().to_file(shapefile_path)
    return shapefile_path


def main() -> None:
    shapefile_path = write_tunnel_shapefile()
    print(f"Tunnel shapefile saved to {shapefile_path}")


if __name__ == "__main__":
    main()
