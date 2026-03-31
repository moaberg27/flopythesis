import os
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

# Tunnelparametrar
TUNNEL_CENTER_DEPTH_M = 80.0
TUNNEL_RADIUS_M = 4.0
TUNNEL_EXTEND_M = 1.0
TUNNEL_SPACING_M = 0.5  # Distance between circular cross-sections
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

# Skapa cirkulär tunnel genom att placera många cirkulära buffers längs linjen
num_circles = int(length / TUNNEL_SPACING_M) + 1
circles = []
for i in range(num_circles):
    t = i / (num_circles - 1) if num_circles > 1 else 0
    x = x0_ext + t * (x1_ext - x0_ext)
    y = y0_ext + t * (y1_ext - y0_ext)
    circles.append(Point(x, y).buffer(TUNNEL_RADIUS_M))

# Slå samman alla cirkel-buffers till en kontinuerlig tunnel
tunnel_poly = unary_union(circles)

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
