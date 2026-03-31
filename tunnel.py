import os
import geopandas as gpd
from shapely.geometry import LineString

# Tunnelparametrar
TUNNEL_CENTER_DEPTH_M = 80.0
TUNNEL_RADIUS_M = 4.0
TUNNEL_EXTEND_M = 1.0
X0, Y0 = 0.0, 50.0
X1, Y1 = 100.0, 50.0

# Skapa folder
folder = "tunneldata"
os.makedirs(folder, exist_ok=True)

# Linje genom hela modellen (forlangd 1 m pa bada sidor)
dx = X1 - X0
dy = Y1 - Y0
length = (dx**2 + dy**2) ** 0.5
ux = dx / length
uy = dy / length
x0_ext = X0 - TUNNEL_EXTEND_M * ux
y0_ext = Y0 - TUNNEL_EXTEND_M * uy
x1_ext = X1 + TUNNEL_EXTEND_M * ux
y1_ext = Y1 + TUNNEL_EXTEND_M * uy
line = LineString([(x0_ext, y0_ext), (x1_ext, y1_ext)])

# Buffra för tunnelradie
tunnel_poly = line.buffer(TUNNEL_RADIUS_M)

# Spara shapefil
gdf = gpd.GeoDataFrame(
    {
        "id": [1],
        "depth_m": [TUNNEL_CENTER_DEPTH_M],
        "radius_m": [TUNNEL_RADIUS_M],
        "x0": [x0_ext],
        "y0": [y0_ext],
        "x1": [x1_ext],
        "y1": [y1_ext],
    },
    geometry=[tunnel_poly],
    crs="EPSG:3006",
)
gdf.to_file(os.path.join(folder, "tunnel.shp"))
print(f"Tunnel shapefile saved to {os.path.join(folder, 'tunnel.shp')}")
