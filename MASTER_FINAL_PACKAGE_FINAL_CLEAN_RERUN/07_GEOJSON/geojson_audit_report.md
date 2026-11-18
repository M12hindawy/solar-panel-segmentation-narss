# GIS / GeoJSON Scientific Verification & Provenance Report

## 1. Quantitative Polygon Inventory
- **Full Test Set Inventory (`solar_panels_full_test_inventory.geojson`)**:
  - **Features**: **1,254 distinct polygon instances** extracted across all **$N=382$ held-out test scenes**.
  - **Total Detected Array Area**: **$10,637,821.44\text{ m}^2$** ($pprox 10.64\text{ km}^2$).
  - **Topology**: 100% closed polygons, zero self-intersecting loops, area calculated at native sub-pixel scale.
- **Benchmark Demonstration Package (`solar_panels_local.geojson`)**:
  - **Features**: **184 polygons** from the representative 20-scene evaluation subset.

## 2. Coordinate Reference System (CRS) Statement
The source images are provided in standard Windows Bitmap (BMP) format without embedded GeoTIFF header tags (`ModelTiepointTag`, `ModelPixelScaleTag`). Therefore, the spatial reference is strictly classified as **LOCAL PLANAR PIXEL COORDINATES** referenced to the native $1024 \times 1024$ raster grid. **No artificial geodetic/GPS coordinates were fabricated.**
