from pathlib import Path
import zipfile
import geopandas as gpd

shz_path = Path("LDNPA_PROW.shz")
output_dir = Path("data/prow")
output_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(shz_path) as zf:
    zf.extractall(output_dir)

shapefile = next(output_dir.glob("*.shp"))

gdf = gpd.read_file(shapefile)

gdf_wgs84 = gdf.to_crs(epsg=4326)
gdf_wgs84.to_file(output_dir / "ldnpa_prow.geojson", driver="GeoJSON")

gdf.to_file(output_dir / "ldnpa_prow.gpkg", driver="GPKG")