# -*- coding: utf-8 -*-
"""
STATIC FLOOD FACTORS  +  OBSERVED FLOOD (KML)  ->  SUPPORT ANALYSIS   (QGIS Python Console)

  A. DEM -> clean -> smooth -> elevation, slope, curvature, TWI (MFD), distance to river,
     drainage density, flow accumulation, SPI, HAND, sink depth
     (+ optional: distance to tanks / roads / canals, built-up, soil, rainfall, land use)
  B. KML flood polygons -> raster on the DEM grid, permanent water (JRC / tanks) masked out
  C. Support, three levels
       1. Descriptive : per class  % area, % of flood, Frequency Ratio (FR)      -> factor_vs_flood.csv + charts
       2. Statistical : correlation + VIF (redundant factors dropped)            -> correlation_vif.csv
       3. Predictive  : logistic regression (+ Random Forest if scikit-learn exists)
                        70/30 split + spatial-block CV, AUC, ROC, importance      -> model_*.csv, roc.jpg, importance.jpg
                        5-class susceptibility map with observed flood outline
  D. JPG maps (Publication + Presentation) and combined panels

Every optional input = None is skipped automatically. Everything is created on the DEM grid,
so nothing outside the DEM region is saved. Temporary files go to <out_root>/_tmp (deleted at the end).

Run in: QGIS > Plugins > Python Console > Show Editor > Open Script > Run
"""
import os, math, csv, heapq, time, shutil
import numpy as np
import processing
from osgeo import gdal, ogr
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsVectorLayer, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform, QgsPalettedRasterRenderer, QgsFillSymbol, QgsLineSymbol,
    QgsPrintLayout, QgsLayoutItemMap, QgsLayoutItemLabel, QgsLayoutItemMapGrid,
    QgsLayoutItemShape, QgsLayoutExporter, QgsLayoutPoint, QgsLayoutSize,
    QgsLayoutMeasurement, QgsUnitTypes
)
from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtCore import Qt

try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SK = True
except Exception:
    HAS_SK = False
try:
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# =====================================================================
# 1. USER SETTINGS
# =====================================================================
dem_path      = r"C:\Users\HP\Desktop\Chennai_shape_final\dem_clipped.tif"
boundary_path = None          # None = basin boundary derived from the DEM itself
river_path    = r"E:\New folder\OneDrive\Pramit_PhD\Chhenai Static\Riv\riv_net_corr.shp"
EXTRA_STREAM_PATHS = []       # optional extra line shapefiles (Buckingham Canal, drains ...)

FLOOD_PATH    = r"E:\New folder\OneDrive\Pramit_PhD\Chhenai Static\NEW Static\flood map.kml"   # <-- EDIT: your KML/KMZ
RAIN_PATH     = r"E:\New folder\OneDrive\Pramit_PhD\Chhenai Static\NC to raster2\2_rain_2015_sum.tif"

JRC_PATH      = None          # JRC Global Surface Water "occurrence" tif (permanent water mask)
LULC_PATH     = None          # ESA WorldCover tif (land use)
BUILTUP_PATH  = None          # GHSL built-up tif
SOIL_PATH     = None          # SoilGrids clay (or sand) tif
ROAD_PATH     = None          # line shapefile (OSM highways)
CANAL_PATH    = None          # line shapefile (OSM canal / drain)
TANK_PATH     = None          # polygon shapefile (tanks / wetlands / ponds)

out_root      = r"E:\New folder\OneDrive\Pramit_PhD\Chhenai Static\New static 2"
STUDY_AREA    = "Chennai"

TARGET_EPSG       = None      # None = auto UTM zone (Chennai -> EPSG:32644)
ASSUME_SOURCE_CRS = "EPSG:4326"
RESAMPLING        = 1         # 0 nearest, 1 bilinear, 2 cubic
WORKING_RES       = None      # metres (e.g. 30) to resample; None = keep

# --- DEM cleaning / smoothing ---
ELEV_MIN = None
ELEV_MAX = None
SMOOTH_SIGMA_CELLS = 2.5      # 0 = off (use ~1 for FABDEM / MERIT)

# --- hydrology ---
STREAM_AREA_KM2   = 0.25
ADD_DEM_STREAMS_TO_DENSITY = True
DD_RADIUS_M       = 2000
USE_MFD_FOR_TWI   = True
MFD_EXPONENT      = 1.1
MFD_MAX_CELLS     = 4_000_000

# --- flood raster ---
FLOOD_OUTLINE_COLOR = "128,0,128,255"   # observed-flood outline drawn on all maps
SHOW_FLOOD_OUTLINE  = True
JRC_OCC_MIN   = 50            # JRC occurrence (%) >= this = permanent water, masked
MASK_TANKS    = True          # tanks / wetlands (TANK_PATH) are masked as permanent water
OUTLINE_MIN_AREA_M2 = 20000   # outline drawn on maps ignores flood islands / holes smaller than this (2 ha)
FLOOD_OUTLINE_WIDTH = 0.2     # mm; the analysis always uses the exact (unsimplified) flood raster

# --- terrain smoothing for slope / curvature (noisy on flat deltas) ---
SLOPE_CURV_SIGMA = 4.0        # Gaussian sigma (cells) used ONLY for slope + curvature; 0 = use SMOOTH_SIGMA_CELLS
USE_MFD_FOR_SPI  = True       # SPI from MFD accumulation (removes the straight N-S streaks of D8 on flats)
RAIN_MIN_EXPECTED_MM = 400    # warn if the regional rainfall total is below this (1 Nov - 5 Dec 2015 should be far higher)
EXPORT_PRESENTATION = True    # False = only the Publication style (faster)

# --- class limits (manual lists; the rest is rounded equal interval / std / log) ---
FLOWACC_BREAKS = [10, 100, 1000, 10000]
HAND_BREAKS    = [0.25, 0.5, 1, 2]       # Chennai is so flat that >1 m is already 'high'
SINK_BREAKS    = [0.05, 0.25, 0.5, 1]
DIST_BREAKS    = [500, 1000, 2000, 5000]       # river
TANK_DIST_BREAKS = [250, 500, 1000, 2000]
ROAD_DIST_BREAKS = [100, 250, 500, 1000]
CANAL_DIST_BREAKS = [250, 500, 1000, 2000]
SUSC_BREAKS    = [0.2, 0.4, 0.6, 0.8]          # susceptibility probability classes

# --- support analysis ---
RAIN_IN_MODEL     = False     # IMD 0.25 deg is ~uniform over a city -> context map only
SAMPLE_SPACING_M  = 90        # minimum spacing between sample points (grid thinning)
N_PER_CLASS       = 5000      # max flooded (= max non-flooded) samples
NONFLOOD_BUFFER_M = 100       # non-flood samples are taken at least this far from the flood edge
TEST_FRAC         = 0.30
BLOCK_M           = 3000      # spatial block size for spatial cross-validation
CV_FOLDS          = 5
VIF_MAX           = 10        # continuous factors above this are dropped one by one
LULC_MIN_SHARE    = 0.02      # land-use classes covering less of the area are not used in the model
USE_RF            = True      # Random Forest too, if scikit-learn is available
SEED              = 42

N_CLASSES    = 5
ANNOT_FORMAT = "decimal"      # "decimal" -> 80.25E   |  "dm" -> 80 15'E
RAMP_A = ["#3f74c4", "#a7b8c9", "#fdf8b0", "#f4a27a", "#d7261e"]   # blue -> red
RIVER_WIDTH = 0.3
RIVER_COLOR = "0,0,0,255"

COMBINED_SETS = {             # file name -> keys, columns, max map height in mm (missing keys are skipped)
    "00_combined_factors": dict(keys=["elevation", "slope", "curvature", "twi", "dist_river", "drain_density"], ncols=3, hmax=70),
    "01_combined_hydro":   dict(keys=["hand", "sinks", "flow_acc", "spi", "builtup", "rain"], ncols=3, hmax=70),
    "02_combined_result":  dict(keys=["flood_extent", "susceptibility", "agreement"], ncols=3, hmax=85),
}

# =====================================================================
# 2. FOLDERS
# =====================================================================
rast_dir = os.path.join(out_root, "rasters")
tmp_dir  = os.path.join(out_root, "_tmp")
tab_dir  = os.path.join(out_root, "tables")
fig_dir  = os.path.join(out_root, "figures")
for d in (out_root, rast_dir, tmp_dir, tab_dir, fig_dir):
    os.makedirs(d, exist_ok=True)
project = QgsProject.instance()
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:6.0f}s] {msg}")

if not (FLOOD_PATH and os.path.exists(FLOOD_PATH)):
    raise Exception("Flood KML not found - edit FLOOD_PATH: " + str(FLOOD_PATH))

# =====================================================================
# 3. GENERIC HELPERS
# =====================================================================
def shifted(a, dy, dx, fill=np.nan):
    """out[i, j] = a[i+dy, j+dx]; outside the array -> fill."""
    h, w = a.shape
    out = np.full(a.shape, fill, dtype=float)
    out[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)] = \
        a[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
    return out

def read_raster(path):
    ds = gdal.Open(path)
    b = ds.GetRasterBand(1)
    a = b.ReadAsArray().astype("float64")
    nd = b.GetNoDataValue()
    if nd is not None:
        a[a == nd] = np.nan
    a[np.abs(a) > 1e30] = np.nan
    return a, ds.GetGeoTransform(), ds.GetProjection()

def write_raster(path, arr, gt, proj, nodata=-9999, gdt=gdal.GDT_Float32):
    ds = gdal.GetDriverByName("GTiff").Create(
        path, arr.shape[1], arr.shape[0], 1, gdt, ["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(gt); ds.SetProjection(proj)
    b = ds.GetRasterBand(1); b.SetNoDataValue(nodata)
    b.WriteArray(np.where(np.isnan(arr), nodata, arr))
    ds.FlushCache(); ds = None

def nice_step(x):
    mag = 10 ** math.floor(math.log10(x))
    return max(m * mag for m in (1, 2, 5, 10) if m * mag <= x)

def nice_ceil(x):
    mag = 10 ** math.floor(math.log10(x))
    return min(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= x)

def fmt(x):
    ax = abs(x)
    if ax >= 1000: return f"{x:,.0f}"
    if ax >= 100:  return f"{x:.0f}"
    if ax >= 10:   return f"{x:.1f}"
    if ax >= 1:    return f"{x:.2f}"
    if ax >= 0.1:  return f"{x:.3f}"
    return f"{x:.4f}"

def gconv(a, sigma):
    """separable Gaussian convolution, zero padding, array must not contain NaN"""
    r = max(1, int(math.ceil(3 * sigma)))
    offs = range(-r, r + 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    tmp = np.zeros(a.shape)
    for o, wt in zip(offs, k):
        tmp += wt * shifted(a, 0, o, fill=0.0)
    out = np.zeros(a.shape)
    for o, wt in zip(offs, k):
        out += wt * shifted(tmp, o, 0, fill=0.0)
    return out

def gauss_smooth(a, sigma):
    """NaN-aware Gaussian smoothing (normalised convolution)"""
    w = (~np.isnan(a)).astype(float)
    num = gconv(np.nan_to_num(a), sigma)
    den = gconv(w, sigma)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    return np.where(np.isnan(a), np.nan, out)

def block_sum(a, f):
    h, w = a.shape
    H, W = -(-h // f) * f, -(-w // f) * f
    p = np.zeros((H, W)); p[:h, :w] = a
    return p.reshape(H // f, f, W // f, f).sum(axis=(1, 3))

def upsample_bilinear(c, f, shape):
    h, w = shape
    yy = np.clip((np.arange(h) + 0.5) / f - 0.5, 0, c.shape[0] - 1)
    xx = np.clip((np.arange(w) + 0.5) / f - 0.5, 0, c.shape[1] - 1)
    y0 = np.floor(yy).astype(int); y1 = np.minimum(y0 + 1, c.shape[0] - 1); fy = (yy - y0)[:, None]
    x0 = np.floor(xx).astype(int); x1 = np.minimum(x0 + 1, c.shape[1] - 1); fx = (xx - x0)[None, :]
    return (c[np.ix_(y0, x0)] * (1 - fy) * (1 - fx) + c[np.ix_(y0, x1)] * (1 - fy) * fx +
            c[np.ix_(y1, x0)] * fy * (1 - fx) + c[np.ix_(y1, x1)] * fy * fx)

def fill_voids(z, bad):
    """replace cells flagged in `bad` by the mean of valid neighbours, growing inwards"""
    a = z.copy(); a[bad] = np.nan
    todo = bad.copy()
    dirs = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
    for _ in range(1000):
        if not todo.any(): break
        st = np.stack([shifted(a, i, j) for i, j in dirs])
        cnt = np.sum(~np.isnan(st), axis=0)
        new = np.where(cnt > 0, np.nansum(st, axis=0) / np.maximum(cnt, 1), np.nan)
        fillm = todo & (cnt > 0)
        if not fillm.any(): break
        a[fillm] = new[fillm]; todo &= ~fillm
    return a

# =====================================================================
# 4. CRS + REPROJECT + CLIP
# =====================================================================
dem = QgsRasterLayer(dem_path, "DEM_original")
if not dem.isValid():
    raise Exception("Cannot load DEM: " + dem_path)
src_crs = dem.crs() if dem.crs().isValid() else QgsCoordinateReferenceSystem(ASSUME_SOURCE_CRS)
if TARGET_EPSG:
    tgt_crs = QgsCoordinateReferenceSystem(TARGET_EPSG)
else:
    c = QgsCoordinateTransform(src_crs, QgsCoordinateReferenceSystem("EPSG:4326"),
                               project).transform(dem.extent().center())
    tgt_crs = QgsCoordinateReferenceSystem(
        f"EPSG:{(32600 if c.y() >= 0 else 32700) + int((c.x() + 180) / 6) + 1}")
tag = tgt_crs.authid().replace(":", "")
log(f"CRS: {src_crs.authid()}  ->  {tgt_crs.authid()} ({tgt_crs.description()})")

tmp_warp = os.path.join(tmp_dir, f"_dem_warp_{tag}.tif")
dem_utm  = os.path.join(rast_dir, f"elevation_raw_{tag}.tif")
processing.run("gdal:warpreproject", {
    "INPUT": dem_path, "SOURCE_CRS": src_crs, "TARGET_CRS": tgt_crs,
    "RESAMPLING": RESAMPLING, "NODATA": -9999, "TARGET_RESOLUTION": WORKING_RES,
    "OPTIONS": "COMPRESS=DEFLATE", "DATA_TYPE": 0, "TARGET_EXTENT": None,
    "MULTITHREADING": True, "EXTRA": "", "OUTPUT": tmp_warp})

def _boundary_layer(path):
    lyr = QgsVectorLayer(path, "Basin boundary", "ogr")
    lyr.renderer().setSymbol(QgsFillSymbol.createSimple({
        "color": "0,0,0,0", "outline_color": "0,0,0,255", "outline_width": "0.5"}))
    project.addMapLayer(lyr)
    return lyr

if boundary_path and os.path.exists(boundary_path):
    bnd_path = os.path.join(rast_dir, f"boundary_{tag}.gpkg")
    processing.run("native:reprojectlayer", {
        "INPUT": QgsVectorLayer(boundary_path, "b", "ogr"),
        "TARGET_CRS": tgt_crs, "OUTPUT": bnd_path})
    processing.run("gdal:cliprasterbymasklayer", {
        "INPUT": tmp_warp, "MASK": bnd_path, "SOURCE_CRS": tgt_crs, "TARGET_CRS": tgt_crs,
        "NODATA": -9999, "ALPHA_BAND": False, "CROP_TO_CUTLINE": True,
        "KEEP_RESOLUTION": True, "SET_RESOLUTION": False, "MULTITHREADING": False,
        "OPTIONS": "COMPRESS=DEFLATE", "DATA_TYPE": 0, "OUTPUT": dem_utm})
    boundary_layer = _boundary_layer(bnd_path)
else:
    arr0, gt0, proj0 = read_raster(tmp_warp)
    vm = ~np.isnan(arr0)
    if not vm.any():
        raise Exception("DEM has no valid cells after reprojection.")
    ri, ci_ = np.flatnonzero(vm.any(axis=1)), np.flatnonzero(vm.any(axis=0))
    r0, r1, c0, c1 = ri[0], ri[-1] + 1, ci_[0], ci_[-1] + 1
    arr0 = arr0[r0:r1, c0:c1]
    gt_crop = (gt0[0] + c0 * gt0[1], gt0[1], 0.0, gt0[3] + r0 * gt0[5], 0.0, gt0[5])
    write_raster(dem_utm, arr0, gt_crop, proj0)

    mask_tif = os.path.join(tmp_dir, "_valid_mask.tif")
    write_raster(mask_tif, np.where(np.isnan(arr0), np.nan, 1.0), gt_crop, proj0,
                 nodata=0, gdt=gdal.GDT_Byte)
    poly_tmp = os.path.join(tmp_dir, "_valid_polygons.gpkg")
    processing.run("gdal:polygonize", {"INPUT": mask_tif, "BAND": 1, "FIELD": "DN",
                                       "EIGHT_CONNECTEDNESS": False, "EXTRA": "", "OUTPUT": poly_tmp})
    bnd_path = os.path.join(out_root, f"basin_boundary_{tag}.shp")
    processing.run("native:dissolve", {"INPUT": poly_tmp, "FIELD": [], "OUTPUT": bnd_path})
    log("basin boundary derived from DEM and saved: " + bnd_path)
    boundary_layer = _boundary_layer(bnd_path)

# =====================================================================
# 5. DEM CLEANING + SMOOTHING
# =====================================================================
Z0, gt, proj = read_raster(dem_utm)
valid0 = ~np.isnan(Z0)
dx, dy = gt[1], -gt[5]
cell_km2 = dx * dy / 1e6
log(f"DEM grid: {Z0.shape[1]} x {Z0.shape[0]} cells, {dx:.1f} m")
log(f"raw elevation: min {np.nanmin(Z0):.1f}  max {np.nanmax(Z0):.1f}")

bad = np.zeros(Z0.shape, bool)
with np.errstate(invalid="ignore"):
    if ELEV_MIN is not None: bad |= valid0 & (Z0 < ELEV_MIN)
    if ELEV_MAX is not None: bad |= valid0 & (Z0 > ELEV_MAX)
log(f"DEM void / outlier cells filled: {int(bad.sum())}")
Z = fill_voids(Z0, bad) if bad.any() else Z0.copy()
valid = ~np.isnan(Z)
Zs = gauss_smooth(Z, SMOOTH_SIGMA_CELLS) if SMOOTH_SIGMA_CELLS > 0 else Z
log(f"clean elevation: min {np.nanmin(Zs):.1f}  max {np.nanmax(Zs):.1f}")
write_raster(os.path.join(rast_dir, f"dem_clean_smooth_{tag}.tif"), Zs, gt, proj)

# ---- helpers that need the DEM grid ----
def reproj(path, name):
    outp = os.path.join(tmp_dir, f"{name}_{tag}.gpkg")
    processing.run("native:reprojectlayer", {"INPUT": QgsVectorLayer(path, "v", "ogr"),
                                             "TARGET_CRS": tgt_crs, "OUTPUT": outp})
    return outp

def rasterize(vec):
    ext_l = QgsRasterLayer(dem_utm).extent()
    outp = os.path.join(tmp_dir, "_rz_tmp.tif")
    processing.run("gdal:rasterize", {
        "INPUT": vec, "BURN": 1, "USE_Z": False, "UNITS": 1, "WIDTH": dx, "HEIGHT": dy,
        "EXTENT": f"{ext_l.xMinimum()},{ext_l.xMaximum()},{ext_l.yMinimum()},{ext_l.yMaximum()} [{tgt_crs.authid()}]",
        "NODATA": 0, "OPTIONS": "", "DATA_TYPE": 0, "INIT": 0, "INVERT": False,
        "EXTRA": "", "OUTPUT": outp})
    a = read_raster(outp)[0]
    if a.shape != Z.shape:
        raise Exception("Rasterised grid differs from DEM grid - check extents.")
    return np.nan_to_num(a) > 0

def proximity(mask, name):
    """distance (m) to the nearest True cell of mask; NaN outside the DEM"""
    if not mask.any():
        return None
    mt = os.path.join(tmp_dir, f"_prox_{name}_in.tif")
    ot = os.path.join(tmp_dir, f"_prox_{name}_out.tif")
    write_raster(mt, np.where(valid, mask.astype(float), np.nan), gt, proj, nodata=255, gdt=gdal.GDT_Byte)
    processing.run("gdal:proximity", {"INPUT": mt, "BAND": 1, "VALUES": "1", "UNITS": 0,
                                      "MAX_DISTANCE": 0, "REPLACE": 0, "NODATA": 0, "OPTIONS": "",
                                      "EXTRA": "", "DATA_TYPE": 5, "OUTPUT": ot})
    d = read_raster(ot)[0]
    d[~valid] = np.nan
    return d

def align(path, alg, name):
    """warp any raster (any CRS / extent) onto the DEM grid; NaN outside the DEM"""
    outp = os.path.join(tmp_dir, f"{name}_aligned.tif")
    srcds = gdal.Open(path)
    if srcds is None:
        raise Exception("Cannot open raster: " + path)
    kw = dict(format="GTiff", dstSRS=proj,
              outputBounds=(gt[0], gt[3] + gt[5] * Z.shape[0], gt[0] + gt[1] * Z.shape[1], gt[3]),
              width=Z.shape[1], height=Z.shape[0], resampleAlg=alg,
              dstNodata=-9999, outputType=gdal.GDT_Float32)
    if not srcds.GetProjection():
        kw["srcSRS"] = ASSUME_SOURCE_CRS
    res = gdal.Warp(outp, srcds, **kw)
    if res is None:
        raise Exception("gdal.Warp failed for " + path)
    res = None
    a = read_raster(outp)[0]
    a[~valid] = np.nan
    return a

# =====================================================================
# 6. TERRAIN + HYDROLOGY (NumPy)
# =====================================================================
def neighbours(z):
    out = {}
    for k, (a, b) in dict(a=(-1, -1), b=(-1, 0), c=(-1, 1), d=(0, -1),
                          f=(0, 1), g=(1, -1), h=(1, 0), i=(1, 1)).items():
        n = shifted(z, a, b)
        out[k] = np.where(np.isnan(n), z, n)
    return out

Zt = gauss_smooth(Z, SLOPE_CURV_SIGMA) if SLOPE_CURV_SIGMA > 0 else Zs
nb = neighbours(Zt)
p = ((nb["c"] + 2 * nb["f"] + nb["i"]) - (nb["a"] + 2 * nb["d"] + nb["g"])) / (8 * dx)
q = ((nb["g"] + 2 * nb["h"] + nb["i"]) - (nb["a"] + 2 * nb["b"] + nb["c"])) / (8 * dy)
slope = np.degrees(np.arctan(np.hypot(p, q)))                      # Horn, degrees
D = ((nb["d"] + nb["f"]) / 2 - Zt) / dx ** 2
E = ((nb["b"] + nb["h"]) / 2 - Zt) / dy ** 2
curv = -2 * (D + E) * 100                                          # Zevenbergen-Thorne, x100
del nb, p, q, D, E
log("slope + curvature done")

def fill_depressions(z):
    """Priority-flood depression filling (with epsilon so flats drain)."""
    zp = np.pad(z, 1, constant_values=np.nan)
    H, W = zp.shape
    vmask = ~np.isnan(zp)
    flat = np.where(vmask, zp, 0.0).ravel().tolist()
    vl = vmask.ravel().tolist()
    inner = np.ones(vmask.shape, bool)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a or b:
                inner &= shifted(vmask.astype(float), a, b) == 1
    seeds = np.flatnonzero((vmask & ~inner).ravel()).tolist()
    closed = bytearray(H * W)
    heap = []
    for s in seeds:
        closed[s] = 1; heap.append((flat[s], s))
    heapq.heapify(heap)
    offs = [-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1]
    nextafter = getattr(math, "nextafter", np.nextafter)
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        e, i = pop(heap)
        for o in offs:
            n = i + o
            if vl[n] and not closed[n]:
                closed[n] = 1
                if flat[n] <= e:
                    flat[n] = nextafter(e, math.inf)
                push(heap, (flat[n], n))
    out = np.array(flat).reshape(H, W)[1:-1, 1:-1]
    return np.where(np.isnan(z), np.nan, out)

def d8_receivers(f):
    h, w = f.shape
    idx = np.arange(h * w).reshape(h, w)
    best = np.zeros((h, w)); recv = np.full((h, w), -1, dtype=np.int64)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if not (a or b): continue
            n = shifted(f, a, b)
            drop = (f - n) / math.hypot(b * dx, a * dy)
            drop = np.where(np.isnan(n), -np.inf, drop)
            better = drop > best
            best = np.where(better, drop, best)
            recv = np.where(better, idx + a * w + b, recv)
    return recv

def flow_accumulation(f, recv):
    ff = f.ravel()
    nvalid = int(np.count_nonzero(~np.isnan(ff)))
    order = np.argsort(-ff, kind="stable")[:nvalid].tolist()
    r = recv.ravel().tolist()
    acc = [1.0] * ff.size
    for i in order:
        j = r[i]
        if j >= 0: acc[j] += acc[i]
    a = np.array(acc).reshape(f.shape)
    a[np.isnan(f)] = np.nan
    return a

def mfd_accumulation(f, pexp):
    """Freeman/Quinn multiple-flow-direction accumulation (cells)"""
    h, w = f.shape
    offs, ws = [], []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if not (a or b): continue
            n = shifted(f, a, b)
            drop = (f - n) / math.hypot(b * dx, a * dy)
            drop = np.where(np.isnan(drop), 0.0, drop)
            ws.append(np.where(drop > 0, drop ** pexp, 0.0)); offs.append(a * w + b)
    tot = np.sum(ws, axis=0); tot[tot == 0] = 1.0
    fr = [(x / tot).ravel().tolist() for x in ws]
    del ws
    ff = f.ravel()
    nvalid = int(np.count_nonzero(~np.isnan(ff)))
    order = np.argsort(-ff, kind="stable")[:nvalid].tolist()
    acc = [1.0] * ff.size
    for i in order:
        ai = acc[i]
        for k in range(8):
            fk = fr[k][i]
            if fk > 0.0:
                acc[i + offs[k]] += ai * fk
    a = np.array(acc).reshape(f.shape)
    a[np.isnan(f)] = np.nan
    return a

log("filling depressions (can take a few minutes for large DEMs) ...")
filled = fill_depressions(Zs)
recv = d8_receivers(filled)
log("D8 flow accumulation ...")
acc = flow_accumulation(filled, recv)
if USE_MFD_FOR_TWI and Z.size <= MFD_MAX_CELLS:
    log("MFD accumulation for TWI ...")
    acc_twi = mfd_accumulation(filled, MFD_EXPONENT)
else:
    if USE_MFD_FOR_TWI:
        log("grid too large for MFD -> using D8 accumulation for TWI")
    acc_twi = acc

tanb = np.maximum(np.tan(np.radians(slope)), 0.001)
twi = np.log(acc_twi * dx / tanb)                # specific catchment area = acc * dx
spi = (acc_twi if USE_MFD_FOR_SPI else acc) * dx * tanb

# ---- streams: DEM streams + river layers ----
thr = max(1.0, STREAM_AREA_KM2 / cell_km2)
streams_dem = (acc >= thr) & valid
log(f"DEM streams: upstream area >= {STREAM_AREA_KM2} km2 ({thr:,.0f} cells), {int(streams_dem.sum())} cells")

river_parts = [pth for pth in [river_path] + list(EXTRA_STREAM_PATHS) if pth and os.path.exists(pth)]
riv_all, river_cells = None, None
if river_parts:
    parts = [reproj(pth, f"_river_part{k}") for k, pth in enumerate(river_parts)]
    riv_all = os.path.join(rast_dir, f"river_all_{tag}.gpkg")
    processing.run("native:mergevectorlayers", {"LAYERS": parts, "CRS": tgt_crs, "OUTPUT": riv_all})
    river_cells = rasterize(riv_all) & valid
    if not river_cells.any():
        raise Exception("River layer does not overlap the DEM.")
    log(f"river cells from shapefiles: {int(river_cells.sum())}")
else:
    log("no river layer found -> DEM-derived streams are used everywhere")

streams_dist = river_cells if river_cells is not None else streams_dem
streams_dd   = (streams_dist | streams_dem) if (ADD_DEM_STREAMS_TO_DENSITY and river_cells is not None) else streams_dist
streams_hand = (streams_dem | river_cells) if river_cells is not None else streams_dem

dist = proximity(streams_dist, "river")

# ---- drainage density ----
sig_cells = (DD_RADIUS_M / 2.0) / dx
fblk = max(1, int(sig_cells / 6))
Sc = block_sum(streams_dd.astype(float), fblk)
Vc = block_sum(valid.astype(float), fblk)
sc = max(0.5, sig_cells / fblk)
ratio = gconv(Sc, sc) / np.maximum(gconv(Vc, sc), 1e-9)
frac = upsample_bilinear(ratio, fblk, Z.shape)
dd = np.where(valid, frac * 1000.0 / dy, np.nan)                   # km / km2

# ---- tanks / wetlands ----
tank_cells, tank_dist = None, None
if TANK_PATH and os.path.exists(TANK_PATH):
    tank_cells = rasterize(reproj(TANK_PATH, "tanks")) & valid
    log(f"tank / wetland cells: {int(tank_cells.sum())}")
    if tank_cells.any():
        tank_dist = proximity(tank_cells, "tank")
    else:
        tank_cells = None
else:
    log("TANK_PATH not set -> tank layers skipped")

# ---- HAND, drains-to-tank, sink depth (walk D8 paths downstream) ----
ff = filled.ravel().tolist()
nvalid = int(np.count_nonzero(~np.isnan(filled)))
order = np.argsort(filled.ravel(), kind="stable")[:nvalid].tolist()      # low -> high
r = recv.ravel().tolist()
st = streams_hand.ravel().tolist()
tk = (tank_cells if tank_cells is not None else np.zeros(Z.shape, bool)).ravel().tolist()
dz = [math.nan] * len(ff)
pt = [False] * len(ff)
for i in order:
    j = r[i]
    dz[i] = ff[i] if (st[i] or j < 0) else dz[j]
    pt[i] = tk[i] or (j >= 0 and pt[j])
hand = np.where(valid, np.maximum(filled - np.array(dz).reshape(Z.shape), 0), np.nan)
drains_tank = np.where(valid, np.array(pt).reshape(Z.shape).astype(float), np.nan)
sink = np.where(valid, filled - Zs, np.nan)                              # depression depth (m)
del ff, order, r, st, tk, dz, pt, filled, recv
log("HAND, sinks, tank connectivity done")

# ---- roads / canals ----
road_dist = canal_dist = None
if ROAD_PATH and os.path.exists(ROAD_PATH):
    road_dist = proximity(rasterize(reproj(ROAD_PATH, "roads")) & valid, "road")
if CANAL_PATH and os.path.exists(CANAL_PATH):
    canal_dist = proximity(rasterize(reproj(CANAL_PATH, "canals")) & valid, "canal")

# ---- extra rasters aligned to the DEM grid ----
rain = builtup = soil = lulc = jrc = None
if RAIN_PATH and os.path.exists(RAIN_PATH):
    rain = align(RAIN_PATH, "bilinear", "rain")
    log(f"rainfall on DEM grid: {np.nanmin(rain):.0f} - {np.nanmax(rain):.0f} mm")
    if np.nanmean(rain) < RAIN_MIN_EXPECTED_MM:
        log(f"  WARNING: mean rainfall {np.nanmean(rain):.0f} mm is far below the expected total for 1 Nov-5 Dec 2015 "
            f"(several hundred to >1000 mm). The NetCDF -> GeoTIFF step is probably wrong: run rain_check.py. "
            f"The rainfall map is context only and is not used in the model.")
if BUILTUP_PATH and os.path.exists(BUILTUP_PATH): builtup = align(BUILTUP_PATH, "bilinear", "builtup")
if SOIL_PATH and os.path.exists(SOIL_PATH):       soil = align(SOIL_PATH, "bilinear", "soil")
if LULC_PATH and os.path.exists(LULC_PATH):       lulc = align(LULC_PATH, "near", "lulc")
if JRC_PATH and os.path.exists(JRC_PATH):         jrc = align(JRC_PATH, "near", "jrc")
log("all static factors computed")

# =====================================================================
# 7. OBSERVED FLOOD (KML) -> raster
# =====================================================================
log("reading flood KML ...")
src_ds = ogr.Open(FLOOD_PATH)
if src_ds is None:
    raise Exception("OGR cannot open the flood file (KML/KMZ driver missing?): " + FLOOD_PATH)
polys = []
for i in range(src_ds.GetLayerCount()):
    nm = src_ds.GetLayerByIndex(i).GetName()
    ql = QgsVectorLayer(f"{FLOOD_PATH}|layername={nm}", nm, "ogr")
    nfeat = ql.featureCount() if ql.isValid() else 0
    log(f"  KML layer '{nm}': valid={ql.isValid()}, features={nfeat}")
    if nfeat == 0:
        continue
    ex = processing.run("native:extractbyexpression", {
        "INPUT": ql, "EXPRESSION": "geometry_type($geometry) = 'Polygon'",
        "OUTPUT": "TEMPORARY_OUTPUT"})["OUTPUT"]
    if ex.featureCount() == 0:
        continue
    rp = processing.run("native:reprojectlayer", {"INPUT": ex, "TARGET_CRS": tgt_crs,
                                                   "OUTPUT": "TEMPORARY_OUTPUT"})["OUTPUT"]
    polys.append(rp)
if not polys:
    raise Exception("No polygon features found in the KML (points / lines only?). See the layer list above.")

fm = os.path.join(tmp_dir, "_flood_merged.gpkg")
fz = os.path.join(tmp_dir, "_flood_noz.gpkg")
ffx = os.path.join(tmp_dir, "_flood_fixed.gpkg")
fds = os.path.join(tmp_dir, "_flood_diss.gpkg")
flood_vec = os.path.join(rast_dir, f"flood_polygons_{tag}.gpkg")
processing.run("native:mergevectorlayers", {"LAYERS": polys, "CRS": tgt_crs, "OUTPUT": fm})
processing.run("native:dropmzvalues", {"INPUT": fm, "DROP_M_VALUES": True, "DROP_Z_VALUES": True, "OUTPUT": fz})
processing.run("native:fixgeometries", {"INPUT": fz, "OUTPUT": ffx})
processing.run("native:dissolve", {"INPUT": ffx, "FIELD": [], "OUTPUT": fds})
processing.run("native:clip", {"INPUT": fds, "OVERLAY": bnd_path, "OUTPUT": flood_vec})
fl_lyr_chk = QgsVectorLayer(flood_vec, "chk", "ogr")
if fl_lyr_chk.featureCount() == 0:
    raise Exception("The KML flood polygons do not overlap the DEM region (check CRS / area).")

# simplified copy used ONLY for drawing the outline (no tiny islands / holes)
_s1 = os.path.join(tmp_dir, "_fo_single.gpkg"); _s2 = os.path.join(tmp_dir, "_fo_big.gpkg")
flood_vec_disp = os.path.join(rast_dir, f"flood_outline_display_{tag}.gpkg")
processing.run("native:multiparttosingleparts", {"INPUT": flood_vec, "OUTPUT": _s1})
processing.run("native:extractbyexpression", {"INPUT": _s1, "EXPRESSION": f"area($geometry) > {OUTLINE_MIN_AREA_M2}", "OUTPUT": _s2})
processing.run("native:deleteholes", {"INPUT": _s2, "MIN_AREA": OUTLINE_MIN_AREA_M2, "OUTPUT": flood_vec_disp})
if QgsVectorLayer(flood_vec_disp, "chk2", "ogr").featureCount() == 0:
    flood_vec_disp = flood_vec

flood_raw = rasterize(flood_vec) & valid

# permanent water (JRC, tanks) is not flood
perm = np.zeros(Z.shape, bool)
if jrc is not None:
    perm |= np.nan_to_num(jrc) >= JRC_OCC_MIN
else:
    log("JRC_PATH not set -> permanent water mask uses tanks only")
if tank_cells is not None and MASK_TANKS:
    perm |= tank_cells
perm &= valid
ok = valid & ~perm                           # analysis mask
flood = flood_raw & ok
n_fl = int(flood.sum())
log(f"flooded cells: {n_fl}  ({n_fl * cell_km2:.2f} km2 = {100 * n_fl / max(1, ok.sum()):.1f}% of the analysis area); "
    f"permanent water masked: {int(perm.sum())} cells")
if n_fl < 100:
    raise Exception("Too few flooded cells after masking - check the KML and the permanent-water mask.")
write_raster(os.path.join(rast_dir, f"flood_observed_{tag}.tif"),
             np.where(valid, flood.astype(float), np.nan), gt, proj, nodata=255, gdt=gdal.GDT_Byte)
dflood = proximity(flood, "flood")           # distance from the flood edge (for non-flood sampling)

# =====================================================================
# 8. CLASSIFICATION + LAYERS + FREQUENCY RATIO
# =====================================================================
def logical_breaks(arr, method, manual=None, k=N_CLASSES):
    """returns the inner break values"""
    v = arr[~np.isnan(arr)]
    lo, hi = float(v.min()), float(v.max())
    if method == "manual":
        inner = sorted(manual)
    elif method == "equal_full":                  # rounded equal interval over the full min-max range
        step = nice_ceil(max(hi - lo, 1e-9) / k)
        start = math.floor(lo / step) * step
        inner = [start + step * i for i in range(1, k)]
    elif method == "equal":                       # rounded equal interval, top class open-ended
        top = float(np.percentile(v, 99))
        if top <= lo: top = hi
        step = nice_ceil(max(top - lo, 1e-9) / k)
        start = math.floor(lo / step) * step
        inner = [start + step * i for i in range(1, k)]
    elif method == "std_zero":                    # symmetric about 0 (curvature)
        s = float(np.std(v))
        b1, b2 = float(f"{0.5 * s:.2g}"), float(f"{1.5 * s:.2g}")
        inner = [-b2, -b1, b1, b2]
    elif method == "log_equal":                   # equal interval in log10, half-decade rounding
        lv = np.log10(np.maximum(v, 1e-12))
        a, b = np.percentile(lv, 1), np.percentile(lv, 99)
        inner = list(10 ** (np.round(np.linspace(a, b, k + 1)[1:-1] * 2) / 2))
    else:
        raise ValueError(method)
    return sorted(set(x for x in inner if lo < x < hi))

def classify(arr, method, manual=None):
    m = ~np.isnan(arr)
    inner = logical_breaks(arr, method, manual)
    cls = np.where(m, np.digitize(arr, inner, right=True) + 1, 0).astype(np.uint8)
    edges = [float(np.nanmin(arr))] + inner + [float(np.nanmax(arr))]
    d = next((d for d in range(4) if all(abs(v - round(v, d)) < 1e-9 for v in inner)), None)
    f = (lambda x: f"{x:,.{d}f}") if d is not None else fmt
    labels = []
    for i in range(len(edges) - 1):
        if i == 0:                  labels.append(f"≤ {f(edges[1])}" if len(edges) > 2 else "all")
        elif i == len(edges) - 2:   labels.append(f"> {f(edges[-2])}")
        else:                       labels.append(f"{f(edges[i])} – {f(edges[i + 1])}")
    return cls, labels

infos, tables, csv_rows = [], {}, []
infos_order_note = []

def add_class_layer(key, title, leg_title, cls, labels, colors, stat=True, outline=True):
    cpath = os.path.join(rast_dir, f"{key}_class_{tag}.tif")
    write_raster(cpath, cls.astype(float), gt, proj, nodata=0, gdt=gdal.GDT_Byte)
    lyr = QgsRasterLayer(cpath, title)
    lyr.setRenderer(QgsPalettedRasterRenderer(lyr.dataProvider(), 1, [
        QgsPalettedRasterRenderer.Class(i + 1, QColor(colors[i]), labels[i]) for i in range(len(labels))]))
    project.addMapLayer(lyr)
    infos.append(dict(key=key, title=title, leg=leg_title, layer=lyr, outline=outline))
    if stat:
        cl = np.where(ok, cls, 0)
        tot = max(1, int(np.count_nonzero(cl)))
        trows = []
        n_nf = max(1, int((ok & ~flood).sum()))
        for i in range(len(labels)):
            m = cl == i + 1
            n, nf = int(m.sum()), int((m & flood).sum())
            nn = n - nf
            pa, pf, pn = n / tot, nf / n_fl, nn / n_nf
            fr = pf / pa if pa > 0 else float("nan")
            lr = pf / pn if pn > 0 else float("nan")          # share of flood / share of non-flood
            trows.append(dict(label=labels[i], pa=100 * pa, pf=100 * pf, pn=100 * pn, fr=fr, lr=lr))
            csv_rows.append([title, i + 1, labels[i], n, round(n * cell_km2, 3), round(100 * pa, 2),
                             round(nf * cell_km2, 3), round(100 * nf / max(n, 1), 2), round(100 * pf, 2),
                             round(100 * pn, 2), round(fr, 3) if fr == fr else "", round(lr, 3) if lr == lr else ""])
        tables[key] = (title, trows)
    log(f"classified {title}: " + " | ".join(labels))

def add_cat_layer(key, title, leg_title, arr, names, colors):
    codes = sorted(int(c) for c in np.unique(arr[~np.isnan(arr)]))
    cls = np.zeros(arr.shape, np.uint8)
    for i, c in enumerate(codes):
        cls[arr == c] = i + 1
    add_class_layer(key, title, leg_title, cls, [names.get(c, f"Class {c}") for c in codes],
                    [colors.get(c, "#999999") for c in codes])
    return codes

# ---- flood extent map (first) ----
fe = np.zeros(Z.shape, np.uint8)
fe[ok] = 1; fe[flood] = 2; fe[perm] = 3
fe_labels = ["Not flooded", "Flooded"] + (["Permanent water (masked)"] if perm.any() else [])
fe_cols = ["#e6e6e6", "#2c7fb8", "#7f7f7f"][:len(fe_labels)]
add_class_layer("flood_extent", "Observed Flood Extent", "KML flood", fe, fe_labels, fe_cols,
                stat=False, outline=False)

# ---- continuous factors ----
FACTORS = [  # key, panel title, legend title, array, method, manual breaks, model-name
    ("elevation",     "Elevation",           "Elevation (m a.s.l.)", Zs,    "manual",   [5, 10, 20, 40], "Elevation"),
    ("slope",         "Slope",               "Slope (°)",            slope, "manual",   [0.25, 0.5, 1, 2], "Slope"),
    ("curvature",     "Curvature",           "Curvature (1/100 m)",  curv,  "std_zero", None, "Curvature"),
    ("twi",           "Topographic Wetness", "TWI",                  twi,   "equal",    None, "TWI"),
    ("dist_river",    "Distance from River", "Distance (m)",         dist,  "manual",   DIST_BREAKS, "Distance to river"),
    ("drain_density", "Drainage Density",    "Density (km/km²)",     dd,    "equal",    None, "Drainage density"),
    ("flow_acc",      "Flow Accumulation",   "Flow acc. (cells)",    acc,   "manual",   FLOWACC_BREAKS, "Flow accumulation (log10)"),
    ("spi",           "Stream Power Index",  "SPI",                  spi,   "log_equal", None, "SPI (log10)"),
    ("hand",          "HAND",                "HAND (m)",             hand,  "manual",   HAND_BREAKS, "HAND"),
    ("sinks",         "Depression Depth",    "Sink depth (m)",       sink,  "manual",   SINK_BREAKS, "Depression depth"),
]
if tank_dist is not None:
    FACTORS.append(("tank_dist", "Distance to Tanks", "Distance (m)", tank_dist, "manual", TANK_DIST_BREAKS, "Distance to tanks"))
if road_dist is not None:
    FACTORS.append(("dist_road", "Distance to Roads", "Distance (m)", road_dist, "manual", ROAD_DIST_BREAKS, "Distance to roads"))
if canal_dist is not None:
    FACTORS.append(("dist_canal", "Distance to Canals", "Distance (m)", canal_dist, "manual", CANAL_DIST_BREAKS, "Distance to canals"))
if builtup is not None:
    FACTORS.append(("builtup", "Built-up Surface", "Built-up (GHSL)", builtup, "equal", None, "Built-up"))
if soil is not None:
    FACTORS.append(("soil", "Soil", "Soil value", soil, "equal", None, "Soil"))
if rain is not None:
    FACTORS.append(("rain", "Rainfall (Nov–Dec 2015)", "Rainfall (mm)", rain, "equal_full", None, "Rainfall"))

TRANSFORM = {"flow_acc": lambda a: np.log10(np.maximum(a, 1.0)),
             "spi": lambda a: np.log10(np.maximum(a, 1e-6))}
feat_names, feat_arrs, feat_kind = [], [], []      # model feature registry

for key, title, leg_title, arr, method, manual, mname in FACTORS:
    write_raster(os.path.join(rast_dir, f"{key}_{tag}.tif"), arr, gt, proj)
    cls, labels = classify(arr, method, manual)
    add_class_layer(key, title, leg_title, cls, labels, RAMP_A)
    if key == "rain" and not RAIN_IN_MODEL:
        continue
    feat_names.append(mname); feat_kind.append("cont")
    feat_arrs.append(TRANSFORM[key](arr) if key in TRANSFORM else arr)

# ---- categorical factors ----
if tank_cells is not None:
    add_cat_layer("drains_tank", "Drains to Tank", "Flow path", drains_tank,
                  {0: "Does not drain to tank", 1: "Drains to tank"}, {0: RAMP_A[0], 1: RAMP_A[4]})
    feat_names.append("Drains to tank"); feat_kind.append("dummy"); feat_arrs.append(drains_tank)

WC_NAMES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
            60: "Bare / sparse", 70: "Snow / ice", 80: "Water", 90: "Wetland", 95: "Mangroves", 100: "Moss / lichen"}
WC_COLS = {10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000", 60: "#b4b4b4",
           70: "#f0f0f0", 80: "#0064c8", 90: "#0096a0", 95: "#00cf75", 100: "#fae6a0"}
if lulc is not None:
    lulc = np.where(np.isnan(lulc), np.nan, np.round(lulc))
    codes = add_cat_layer("lulc", "Land Use / Land Cover", "Land cover", lulc, WC_NAMES, WC_COLS)
    shares = {c: float(np.count_nonzero((lulc == c) & ok)) / max(1, ok.sum()) for c in codes}
    ref = max(shares, key=shares.get)
    for c in codes:
        if c != ref and shares[c] >= LULC_MIN_SHARE:
            feat_names.append("LULC: " + WC_NAMES.get(c, str(c))); feat_kind.append("dummy")
            feat_arrs.append(np.where(valid, (lulc == c).astype(float), np.nan))
    log(f"land-use dummies use '{WC_NAMES.get(ref, ref)}' as reference class")

def write_factor_csv():
    with open(os.path.join(tab_dir, "factor_vs_flood.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Factor", "Class", "Label", "Cells", "Area_km2", "Pct_of_area", "Flooded_km2",
                    "Pct_of_class_flooded", "Pct_of_total_flood", "Pct_of_total_nonflood",
                    "Frequency_ratio", "Likelihood_ratio_flood_vs_nonflood"])
        w.writerows(csv_rows)
write_factor_csv()

# =====================================================================
# 9. SUPPORT ANALYSIS: sampling, correlation, VIF, models, validation
# =====================================================================
# ---- STATS-BEGIN ----
def fit_logit(X, y, l2=1e-3, iters=60):
    A = np.c_[np.ones(len(X)), X]
    b = np.zeros(A.shape[1])
    R = l2 * np.eye(A.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-np.clip(A @ b, -30, 30)))
        W = pr * (1 - pr) + 1e-9
        g = A.T @ (y - pr) - R @ b
        H = (A * W[:, None]).T @ A + R
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-8: break
    pr = 1 / (1 + np.exp(-np.clip(A @ b, -30, 30)))
    W = pr * (1 - pr) + 1e-9
    H = (A * W[:, None]).T @ A + R
    return b, np.linalg.inv(H)

def pred_logit(b, X):
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), X] @ b, -30, 30)))

def auc_score(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = avg_rank[inv]
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def roc_curve(y, s):
    y = np.asarray(y).astype(int)
    o = np.argsort(-np.asarray(s), kind="mergesort"); ys = y[o]
    tps, fps = np.cumsum(ys), np.cumsum(1 - ys)
    return np.r_[0, fps / fps[-1]], np.r_[0, tps / tps[-1]]

def class_metrics(y, s, thr=0.5):
    y = np.asarray(y).astype(int); pr = (np.asarray(s) >= thr).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); tn = int(((pr == 0) & (y == 0)).sum())
    fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
    return dict(acc=(tp + tn) / max(1, len(y)), sens=tp / max(1, tp + fn), spec=tn / max(1, tn + fp))

def vif_all(X):
    out = []
    for j in range(X.shape[1]):
        A = np.c_[np.ones(len(X)), np.delete(X, j, axis=1)]
        beta = np.linalg.lstsq(A, X[:, j], rcond=None)[0]
        res = X[:, j] - A @ beta
        r2 = 1 - res.var() / max(X[:, j].var(), 1e-12)
        out.append(1 / max(1 - r2, 1e-6))
    return np.array(out)

def train(kind, X, y, seed=42):
    """returns predict(X_raw) -> probability, and the fitted internals"""
    if kind == "logit":
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
        b, cov = fit_logit((X - mu) / sd, y)
        return (lambda Z: pred_logit(b, (Z - mu) / sd)), (b, cov)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed).fit(X, y)
    return (lambda Z: rf.predict_proba(Z)[:, 1]), rf

def perm_importance(predict, X, y, rng, repeats=10):
    base = auc_score(y, predict(X))
    out = []
    for j in range(X.shape[1]):
        drops = []
        for _ in range(repeats):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - auc_score(y, predict(Xp)))
        out.append(float(np.mean(drops)))
    return np.array(out)
def best_threshold(y, s):
    """probability threshold maximising Youden's J (sensitivity + specificity - 1)"""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    cands = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
    best_t, best_j = 0.5, -1.0
    for t in cands:
        pr = s >= t
        j = pr[y == 1].mean() - pr[y == 0].mean()
        if j > best_j: best_j, best_t = j, float(t)
    return best_t
# ---- STATS-END ----

rng = np.random.default_rng(SEED)
fin = ok.copy()
for a_ in feat_arrs:
    fin &= ~np.isnan(a_)
stride = max(1, int(round(SAMPLE_SPACING_M / dx)))
grid = np.zeros(Z.shape, bool); grid[::stride, ::stride] = True
pos = np.argwhere(fin & grid & flood)
neg = np.argwhere(fin & grid & ~flood & (np.nan_to_num(dflood, nan=0) >= NONFLOOD_BUFFER_M))
n_s = min(len(pos), len(neg), N_PER_CLASS)
log(f"candidate samples: flooded {len(pos)}, non-flooded {len(neg)} -> using {n_s} of each")
if n_s < 30:
    raise Exception("Too few samples - reduce SAMPLE_SPACING_M / NONFLOOD_BUFFER_M or check the flood layer.")
rc = np.vstack([pos[rng.choice(len(pos), n_s, replace=False)], neg[rng.choice(len(neg), n_s, replace=False)]])
y = np.r_[np.ones(n_s), np.zeros(n_s)]
X = np.column_stack([a_[rc[:, 0], rc[:, 1]] for a_ in feat_arrs])
nf = X.shape[1]

# ---- correlation + VIF ----
Xs = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
corr = [float(np.corrcoef(X[:, j], y)[0, 1]) if X[:, j].std() > 0 else 0.0 for j in range(nf)]
cont = [j for j in range(nf) if feat_kind[j] == "cont"]
vif0 = np.full(nf, np.nan); vif0[cont] = vif_all(Xs[:, cont])
keep = list(cont); dropped = []
while len(keep) > 2:
    v = vif_all(Xs[:, keep])
    if v.max() <= VIF_MAX: break
    j = keep[int(np.argmax(v))]
    dropped.append(j); keep.remove(j)
    log(f"  VIF > {VIF_MAX}: dropped {feat_names[j]} (VIF {v.max():.1f})")
vif1 = np.full(nf, np.nan); vif1[keep] = vif_all(Xs[:, keep])
sel = sorted(keep + [j for j in range(nf) if feat_kind[j] == "dummy"])
with open(os.path.join(tab_dir, "correlation_vif.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Factor", "Correlation_with_flood", "VIF_initial", "VIF_final", "Used_in_model"])
    for j in range(nf):
        w.writerow([feat_names[j], round(corr[j], 3),
                    "" if np.isnan(vif0[j]) else round(vif0[j], 2),
                    "" if np.isnan(vif1[j]) else round(vif1[j], 2), "yes" if j in sel else "no (VIF)"])
names_m = [feat_names[j] for j in sel]
Xm = X[:, sel]
log("model factors: " + ", ".join(names_m))

# ---- splits ----
models = ["logit"] + (["rf"] if (USE_RF and HAS_SK) else [])
if USE_RF and not HAS_SK:
    log("scikit-learn not found -> Random Forest skipped (logistic regression only)")
idx = rng.permutation(len(y)); nt = int(len(y) * TEST_FRAC)
te, tr = idx[:nt], idx[nt:]
bs = max(1, int(round(BLOCK_M / dx)))
blk = (rc[:, 0] // bs).astype(np.int64) * 100000 + (rc[:, 1] // bs)
ublk = rng.permutation(np.unique(blk))
kf = min(CV_FOLDS, len(ublk))
fold_of = {b_: i % kf for i, b_ in enumerate(ublk)} if kf >= 3 else None
fold = np.array([fold_of[b_] for b_ in blk]) if fold_of else None
if fold is None:
    log("too few spatial blocks -> spatial CV skipped (reduce BLOCK_M)")

metric_rows, imp_rows, oof_store, test_store = [], [], {}, {}
mean_cv = {}
for kind in models:
    lab = "Logistic regression" if kind == "logit" else "Random Forest"
    f_, _ = train(kind, Xm[tr], y[tr], SEED)
    ps = f_(Xm[te])
    cm = class_metrics(y[te], ps)
    metric_rows.append([lab, f"Random {int((1 - TEST_FRAC) * 100)}/{int(TEST_FRAC * 100)} test",
                        round(auc_score(y[te], ps), 3), round(cm["acc"], 3), round(cm["sens"], 3),
                        round(cm["spec"], 3), len(te)])
    test_store[kind] = (y[te], ps)
    pi = perm_importance(f_, Xm[te], y[te], rng)
    for nme, v in zip(names_m, pi):
        imp_rows.append([lab, nme, round(float(v), 4)])
    if fold is not None:
        oof = np.full(len(y), np.nan); aucs = []
        for k in range(kf):
            trn, tst = fold != k, fold == k
            if len(np.unique(y[tst])) < 2 or len(np.unique(y[trn])) < 2: continue
            fk, _ = train(kind, Xm[trn], y[trn], SEED)
            oof[tst] = fk(Xm[tst])
            aucs.append(auc_score(y[tst], oof[tst]))
        okm = ~np.isnan(oof)
        if okm.any() and aucs:
            cm = class_metrics(y[okm], oof[okm])
            metric_rows.append([lab, f"Spatial block CV ({len(aucs)} folds, {BLOCK_M / 1000:g} km blocks)",
                                f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}", round(cm["acc"], 3),
                                round(cm["sens"], 3), round(cm["spec"], 3), int(okm.sum())])
            oof_store[kind] = (y[okm], oof[okm]); mean_cv[kind] = float(np.mean(aucs))
for row in metric_rows:
    log("  " + " | ".join(str(x) for x in row))

# ---- final models on all samples ----
f_lr, (b_lr, cov_lr) = train("logit", Xm, y, SEED)
se = np.sqrt(np.diag(cov_lr)); zst = b_lr / se
with open(os.path.join(tab_dir, "model_coefficients.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Term", "Std_coefficient", "Std_error", "z", "p_value", "Odds_ratio_per_SD"])
    for nme, bb, ss, zz in zip(["Intercept"] + names_m, b_lr, se, zst):
        w.writerow([nme, round(float(bb), 4), round(float(ss), 4), round(float(zz), 2),
                    f"{math.erfc(abs(float(zz)) / math.sqrt(2)):.3g}", round(float(math.exp(np.clip(bb, -30, 30))), 3)])
with open(os.path.join(tab_dir, "model_metrics.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Evaluation", "AUC", "Accuracy", "Sensitivity", "Specificity", "N"])
    w.writerows(metric_rows)
with open(os.path.join(tab_dir, "permutation_importance.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Factor", "AUC_drop_when_permuted"])
    w.writerows(imp_rows)

best = max(mean_cv, key=mean_cv.get) if mean_cv else "logit"
log(f"model used for the susceptibility map: {'Random Forest' if best == 'rf' else 'Logistic regression'}")
f_best = f_lr if best == "logit" else train("rf", Xm, y, SEED)[0]

# ---- susceptibility raster ----
pm = ok & fin
ii_all = np.flatnonzero(pm.ravel())
cols = [feat_arrs[j].ravel() for j in sel]
prob = np.full(Z.size, np.nan)
for s0 in range(0, len(ii_all), 500000):
    ii = ii_all[s0:s0 + 500000]
    prob[ii] = f_best(np.column_stack([c_[ii] for c_ in cols]))
prob = prob.reshape(Z.shape)
write_raster(os.path.join(rast_dir, f"susceptibility_prob_{tag}.tif"), prob, gt, proj)
auc_full = auc_score(flood[pm].astype(int), prob[pm])
log(f"AUC of susceptibility vs the full observed flood raster: {auc_full:.3f} (includes training cells)")

susc_cls = np.where(np.isnan(prob), 0, np.digitize(np.nan_to_num(prob), SUSC_BREAKS, right=True) + 1).astype(np.uint8)
susc_labels = ["Very low", "Low", "Moderate", "High", "Very high"]
add_class_layer("susceptibility", "Flood Susceptibility", "Susceptibility", susc_cls, susc_labels, RAMP_A)
tt = tables["susceptibility"][1]
cap = tt[3]["pf"] + tt[4]["pf"]; ar = tt[3]["pa"] + tt[4]["pa"]
log(f"High + Very high susceptibility: {ar:.1f}% of the area captures {cap:.1f}% of the observed flood")
with open(os.path.join(tab_dir, "susceptibility_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Class", "Pct_of_area", "Pct_of_observed_flood", "Frequency_ratio"])
    for lab_, r_ in zip(susc_labels, tt):
        w.writerow([lab_, round(r_["pa"], 2), round(r_["pf"], 2), round(r_["fr"], 3)])
    w.writerow([]); w.writerow(["AUC full extent", round(auc_full, 3)])
    w.writerow(["High+Very high: % of area", round(ar, 2)]); w.writerow(["High+Very high: % of flood", round(cap, 2)])

# ---- agreement map: observed vs predicted flood (threshold from out-of-fold / test predictions, Youden J) ----
yy_t, ss_t = oof_store[best] if best in oof_store else test_store[best]
thr_best = best_threshold(yy_t, ss_t)
pred = (prob >= thr_best) & pm
tp = int((pred & flood).sum()); fp = int((pred & ~flood).sum())
fn = int((pm & ~pred & flood).sum()); tn = int((pm & ~pred & ~flood).sum())
nn_ = tp + fp + fn + tn
agree_metrics = [
    ("Threshold (Youden J on held-out predictions)", round(float(thr_best), 3)),
    ("Hits (flooded, predicted) cells", tp), ("Misses (flooded, not predicted) cells", fn),
    ("False alarms (dry, predicted) cells", fp), ("Correct dry cells", tn),
    ("Accuracy", round((tp + tn) / max(1, nn_), 3)),
    ("Hit rate / recall", round(tp / max(1, tp + fn), 3)),
    ("Specificity", round(tn / max(1, tn + fp), 3)),
    ("Precision", round(tp / max(1, tp + fp), 3)),
    ("False alarm ratio", round(fp / max(1, tp + fp), 3)),
    ("Critical success index (CSI)", round(tp / max(1, tp + fn + fp), 3)),
    ("F1", round(2 * tp / max(1, 2 * tp + fp + fn), 3))]
with open(os.path.join(tab_dir, "agreement_metrics.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["Metric", "Value"]); w.writerows(agree_metrics)
for k_, v_ in agree_metrics: log(f"  agreement | {k_}: {v_}")
ag = np.zeros(Z.shape, np.uint8)
ag[pm & ~pred & ~flood] = 1; ag[pred & flood] = 2; ag[pm & ~pred & flood] = 3; ag[pred & ~flood] = 4
add_class_layer("agreement", "Predicted vs Observed Flood", "Agreement", ag,
                ["Correct dry", "Hit (flooded, predicted)", "Miss (flooded, not predicted)", "False alarm (dry, predicted)"],
                ["#e6e6e6", "#2c7fb8", "#d7261e", "#fdae61"], stat=False, outline=False)
write_factor_csv()

# =====================================================================
# 10. CHARTS (matplotlib)
# =====================================================================
def save_fig(fig, base):
    FigureCanvasAgg(fig)
    for ext in ("jpg", "png"):
        try:
            fig.savefig(os.path.join(fig_dir, f"{base}.{ext}"), dpi=300, facecolor="white", bbox_inches="tight")
            log(f"✔ figures/{base}.{ext}"); return
        except Exception as e:
            log(f"  could not save {base}.{ext}: {e}")

def table_chart(kind):
    keys = list(tables); n = len(keys); nc = 4; nr = math.ceil(n / nc)
    fig = Figure(figsize=(nc * 3.4, nr * 3.0))
    axs = fig.subplots(nr, nc, squeeze=False)
    for ax in axs.ravel(): ax.set_visible(False)
    for ax, k in zip(axs.ravel(), keys):
        ax.set_visible(True)
        title, rws = tables[k]; xs = np.arange(len(rws))
        if kind in ("fr", "lr"):
            raw = [r[kind] if r[kind] == r[kind] else 0 for r in rws]
            vals = [min(v, 10) for v in raw]
            ax.bar(xs, vals, color=["#d7261e" if v > 1 else "#3f74c4" for v in vals], edgecolor="k", linewidth=0.3)
            ax.axhline(1, ls="--", lw=0.8, color="k")
            ax.set_ylabel("Frequency ratio" if kind == "fr" else "Likelihood ratio (flood / non-flood)", fontsize=6.5)
        elif kind == "share_area":
            ax.bar(xs - 0.2, [r["pa"] for r in rws], 0.4, color="#a7b8c9", edgecolor="k", linewidth=0.3, label="% of area")
            ax.bar(xs + 0.2, [r["pf"] for r in rws], 0.4, color="#2c7fb8", edgecolor="k", linewidth=0.3, label="% of flood")
            ax.set_ylabel("%", fontsize=7)
            if k == keys[0]: ax.legend(fontsize=6, frameon=False)
        else:
            ax.bar(xs - 0.2, [r["pn"] for r in rws], 0.4, color="#e0a050", edgecolor="k", linewidth=0.3, label="% of non-flood")
            ax.bar(xs + 0.2, [r["pf"] for r in rws], 0.4, color="#2c7fb8", edgecolor="k", linewidth=0.3, label="% of flood")
            ax.set_ylabel("%", fontsize=7)
            if k == keys[0]: ax.legend(fontsize=6, frameon=False)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xticks(xs); ax.set_xticklabels([r["label"] for r in rws], rotation=45, ha="right", fontsize=5.5)
        ax.tick_params(axis="y", labelsize=6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save_fig(fig, {"fr": "FR_all_factors", "lr": "LR_flood_vs_nonflood", "share_area": "share_area_vs_flood",
                   "share_nonflood": "share_flood_vs_nonflood"}[kind])

def roc_chart():
    fig = Figure(figsize=(4.5, 4.5)); ax = fig.subplots()
    cols_ = {"logit": "#3f74c4", "rf": "#d7261e"}
    nm = {"logit": "Logistic", "rf": "Random Forest"}
    for kind, (yy, ss) in test_store.items():
        fp, tp = roc_curve(yy, ss)
        ax.plot(fp, tp, color=cols_[kind], lw=1.6, label=f"{nm[kind]} test (AUC {auc_score(yy, ss):.3f})")
    for kind, (yy, ss) in oof_store.items():
        fp, tp = roc_curve(yy, ss)
        ax.plot(fp, tp, color=cols_[kind], lw=1.2, ls="--", label=f"{nm[kind]} spatial CV (AUC {auc_score(yy, ss):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.set_aspect("equal")
    fig.tight_layout(); save_fig(fig, "roc")

def importance_chart():
    labs = sorted(set(r[0] for r in imp_rows))
    fig = Figure(figsize=(4.2 * len(labs), 0.38 * len(names_m) + 1.4))
    axs = fig.subplots(1, len(labs), squeeze=False)
    for ax, lab in zip(axs.ravel(), labs):
        vals = np.array([r[2] for r in imp_rows if r[0] == lab]); o = np.argsort(vals)
        ax.barh(np.array(names_m)[o], vals[o], color="#3f74c4", edgecolor="k", linewidth=0.3)
        ax.set_title(lab, fontsize=9, fontweight="bold"); ax.set_xlabel("AUC drop when permuted", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); save_fig(fig, "importance")

if HAS_MPL:
    try:
        for _k in ("fr", "lr", "share_area", "share_nonflood"): table_chart(_k)
        roc_chart(); importance_chart()
    except Exception as e:
        log(f"  chart error (maps and tables are unaffected): {e}")
else:
    log("matplotlib not available -> charts skipped (CSV tables are written)")

# =====================================================================
# 11. MAP STYLES  -  title | map + river + flood outline | legend OUTSIDE (right)
# =====================================================================
PUB = dict(
    name="Publication_ready", font="Times New Roman", dpi=600,
    W=174.0, H=None, hmax=150, margin=5, gap=4,
    title=12, ann=8, legt=9, leg=8,
    ann_l=12, ann_b=7, leg_w=46, sw=(8, 5), tick=1.2)

PRES = dict(
    name="Presentation", font="Arial", dpi=300,
    W=338.67, H=190.5, hmax=None, margin=8, gap=8,
    title=30, ann=16, legt=18, leg=16,
    ann_l=26, ann_b=13, leg_w=100, sw=(14, 8.5), tick=2.2)

MM  = QgsUnitTypes.LayoutMillimeters
WGS = QgsCoordinateReferenceSystem("EPSG:4326")
ANNOT = {"decimal": QgsLayoutItemMapGrid.DecimalWithSuffix, "dm": QgsLayoutItemMapGrid.DegreeMinute}
PT = 0.3528   # 1 pt in mm

river_layer = None
if riv_all:
    rclip = os.path.join(rast_dir, f"river_clipped_{tag}.gpkg")
    processing.run("native:clip", {"INPUT": riv_all, "OVERLAY": bnd_path, "OUTPUT": rclip})
    river_layer = QgsVectorLayer(rclip, "Rivers", "ogr")
    river_layer.renderer().setSymbol(QgsLineSymbol.createSimple(
        {"color": RIVER_COLOR, "width": str(RIVER_WIDTH), "capstyle": "round", "joinstyle": "round"}))
    project.addMapLayer(river_layer)

flood_layer = None
if SHOW_FLOOD_OUTLINE:
    flood_layer = QgsVectorLayer(flood_vec_disp, "Observed flood", "ogr")
    flood_layer.renderer().setSymbol(QgsFillSymbol.createSimple(
        {"color": "0,0,0,0", "outline_color": FLOOD_OUTLINE_COLOR, "outline_width": str(FLOOD_OUTLINE_WIDTH)}))
    project.addMapLayer(flood_layer)

def qfont(style, size, bold=False, italic=False):
    f = QFont(style["font"]); f.setPointSizeF(size); f.setBold(bold); f.setItalic(italic)
    return f

def add_label(lay, text, x, y, w, h, font, halign=Qt.AlignLeft, valign=Qt.AlignVCenter):
    lb = QgsLayoutItemLabel(lay)
    lb.setText(text); lb.setFont(font)
    lb.setHAlign(halign); lb.setVAlign(valign)
    lb.attemptMove(QgsLayoutPoint(x, y, MM)); lb.attemptResize(QgsLayoutSize(w, h, MM))
    lay.addLayoutItem(lb)

def fit(asp, bw, bh):
    return (bw, bw * asp) if bw * asp <= bh else (bh / asp, bh)

def title_h(S):
    return S["title"] * PT * 1.9

def geom(info, S, w, h):
    ext = info["layer"].extent()
    bw = w - S["ann_l"] - S["gap"] - S["leg_w"]
    bh = h - title_h(S) - S["ann_b"]
    return fit(ext.height() / ext.width(), bw, bh)

def _get(c, name):
    v = getattr(c, name)
    return v() if callable(v) else v

OVERLAY_ON = True      # switched per map set in the export loop
SET_TAG = "set"
def has_outline(info):
    return OVERLAY_ON and bool(flood_layer) and info.get("outline", True)

def add_map(lay, info, mx, my, mw, mh, S):
    lyr = info["layer"]; ext = lyr.extent()
    m = QgsLayoutItemMap(lay)
    m.attemptMove(QgsLayoutPoint(mx, my, MM)); m.attemptResize(QgsLayoutSize(mw, mh, MM))
    m.setCrs(tgt_crs)
    m.setLayers(([flood_layer] if has_outline(info) else []) + ([river_layer] if river_layer else []) +
                [boundary_layer, lyr])
    m.zoomToExtent(ext)
    m.setBackgroundEnabled(True); m.setBackgroundColor(QColor(255, 255, 255))
    m.setFrameEnabled(True); m.setFrameStrokeWidth(QgsLayoutMeasurement(0.25))
    m.setFrameStrokeColor(QColor(0, 0, 0))
    lay.addLayoutItem(m)

    bb = QgsCoordinateTransform(tgt_crs, WGS, project).transformBoundingBox(ext)
    step = nice_step(max(bb.width(), bb.height()) / 3)
    g = QgsLayoutItemMapGrid("graticule", m)
    m.grids().addGrid(g)
    g.setCrs(WGS); g.setUnits(QgsLayoutItemMapGrid.MapUnit)
    g.setIntervalX(step); g.setIntervalY(step)
    g.setStyle(QgsLayoutItemMapGrid.FrameAnnotationsOnly)
    g.setFrameStyle(QgsLayoutItemMapGrid.InteriorTicks)
    g.setFrameWidth(S["tick"])
    g.setFramePenSize(0.2); g.setFramePenColor(QColor(0, 0, 0))
    g.setAnnotationEnabled(True)
    g.setAnnotationFont(qfont(S, S["ann"])); g.setAnnotationFrameDistance(1.5)
    g.setAnnotationFormat(ANNOT[ANNOT_FORMAT])
    g.setAnnotationPrecision(max(1, math.ceil(-math.log10(step))))
    for side, flag in {QgsLayoutItemMapGrid.Left: QgsLayoutItemMapGrid.FrameLeft,
                       QgsLayoutItemMapGrid.Right: QgsLayoutItemMapGrid.FrameRight,
                       QgsLayoutItemMapGrid.Top: QgsLayoutItemMapGrid.FrameTop,
                       QgsLayoutItemMapGrid.Bottom: QgsLayoutItemMapGrid.FrameBottom}.items():
        g.setAnnotationPosition(QgsLayoutItemMapGrid.OutsideMapFrame, side)
        g.setFrameSideFlag(flag, True)
    g.setAnnotationDirection(QgsLayoutItemMapGrid.Horizontal, QgsLayoutItemMapGrid.Left)
    g.setAnnotationDisplay(QgsLayoutItemMapGrid.LatitudeOnly,  QgsLayoutItemMapGrid.Left)
    g.setAnnotationDisplay(QgsLayoutItemMapGrid.LongitudeOnly, QgsLayoutItemMapGrid.Bottom)
    g.setAnnotationDisplay(QgsLayoutItemMapGrid.HideAll,       QgsLayoutItemMapGrid.Right)
    g.setAnnotationDisplay(QgsLayoutItemMapGrid.HideAll,       QgsLayoutItemMapGrid.Top)
    return m

def _box(lay, x, y, w, h, fill, outline="0,0,0,255", ow="0.15"):
    box = QgsLayoutItemShape(lay)
    box.setShapeType(QgsLayoutItemShape.Rectangle)
    box.setSymbol(QgsFillSymbol.createSimple({"color": fill, "outline_color": outline, "outline_width": ow}))
    box.attemptMove(QgsLayoutPoint(x, y, MM)); box.attemptResize(QgsLayoutSize(w, h, MM))
    lay.addLayoutItem(box)

def row_height(S):
    return max(S["sw"][1], S["leg"] * PT * 1.5) + S["leg"] * PT * 0.6

def add_legend(lay, info, x, y, S):
    sw_w, sw_h = S["sw"]
    t_h = S["legt"] * PT * 2.9
    add_label(lay, info["leg"], x, y, S["leg_w"], t_h, qfont(S, S["legt"], bold=True),
              Qt.AlignLeft, Qt.AlignBottom)
    row_h = row_height(S)
    yy = y + t_h + 2
    for c in info["layer"].renderer().classes():
        col = _get(c, "color"); lab = _get(c, "label")
        _box(lay, x, yy + (row_h - sw_h) / 2, sw_w, sw_h, f"{col.red()},{col.green()},{col.blue()},255")
        add_label(lay, lab, x + sw_w + 2, yy, S["leg_w"] - sw_w - 2, row_h, qfont(S, S["leg"]))
        yy += row_h
    if has_outline(info):
        _box(lay, x, yy + (row_h - sw_h) / 2, sw_w, sw_h, "0,0,0,0", FLOOD_OUTLINE_COLOR, "0.5")
        add_label(lay, "Observed flood", x + sw_w + 2, yy, S["leg_w"] - sw_w - 2, row_h, qfont(S, S["leg"]))

def new_layout(name, w, h, dpi):
    mgr = project.layoutManager()
    old = mgr.layoutByName(name)
    if old: mgr.removeLayout(old)
    lay = QgsPrintLayout(project)
    lay.initializeDefaults(); lay.setName(name)
    lay.pageCollection().page(0).setPageSize(QgsLayoutSize(w, h, MM))
    lay.renderContext().setDpi(dpi)
    mgr.addLayout(lay)
    return lay

def single_layout(info, idx, S):
    W, mg = S["W"], S["margin"]
    th = title_h(S)
    if S["H"] is None:
        _, mh = geom(info, S, W - 2 * mg, th + S["hmax"] + S["ann_b"])
        H = 2 * mg + th + mh + S["ann_b"]
    else:
        H = S["H"]
    lay = new_layout(f"{SET_TAG}_{S['name']}_{info['key']}", W, H, S["dpi"])
    pw, ph = W - 2 * mg, H - 2 * mg
    mw, mh = geom(info, S, pw, ph)
    gw = S["ann_l"] + mw + S["gap"] + S["leg_w"]
    gx = mg + (pw - gw) / 2
    mx = gx + S["ann_l"]
    my = mg + th + (ph - th - S["ann_b"] - mh) / 2
    add_label(lay, f"({chr(97 + idx)}) {info['title']}", mx, my - th, mw + S["gap"] + S["leg_w"], th,
              qfont(S, S["title"], bold=True), Qt.AlignLeft, Qt.AlignVCenter)
    add_map(lay, info, mx, my, mw, mh, S)
    add_legend(lay, info, mx + mw + S["gap"], my, S)
    return lay

def combined_layout(name, chosen, S0, ncols=3, map_hmax=60):
    """grid of panels; each panel = title, map, legend underneath"""
    S = dict(S0, title=9, ann=6, legt=7, leg=6.5, ann_l=11, ann_b=5, sw=(6, 3.5), tick=0.8)
    mg, W = S0["margin"], S0["W"]
    nrows = math.ceil(len(chosen) / ncols)
    cw = (W - 2 * mg) / ncols
    S["leg_w"] = cw - S["ann_l"] - 3
    th = title_h(S)
    ext = chosen[0]["layer"].extent()
    mw, mh = fit(ext.height() / ext.width(), cw - S["ann_l"] - 3, map_hmax)
    nrow_leg = max(len(i["layer"].renderer().classes()) + (1 if has_outline(i) else 0) for i in chosen)
    leg_h = S["legt"] * PT * 2.9 + 2 + nrow_leg * row_height(S)
    ph = th + mh + S["ann_b"] + leg_h + 3
    H = 2 * mg + nrows * ph
    lay = new_layout(name, W, H, S0["dpi"])
    for i, info in enumerate(chosen):
        r_, c_ = divmod(i, ncols)
        x0, y0 = mg + c_ * cw, mg + r_ * ph
        mx, my = x0 + S["ann_l"], y0 + th
        add_label(lay, f"({chr(97 + i)}) {info['title']}", mx, y0, cw - S["ann_l"], th,
                  qfont(S, S["title"], bold=True))
        add_map(lay, info, mx, my, mw, mh, S)
        add_legend(lay, info, mx, my + mh + S["ann_b"], S)
    return lay

def export(lay, folder, base, S):
    s = QgsLayoutExporter.ImageExportSettings(); s.dpi = S["dpi"]
    res = QgsLayoutExporter(lay).exportToImage(os.path.join(folder, base + ".jpg"), s)
    ok_ = res == QgsLayoutExporter.Success
    log(("✔ " if ok_ else "✘ ") + f"{S['name']}/{base}.jpg" + ("" if ok_ else f"  code={res}"))

# =====================================================================
# 12. EXPORT MAPS  -  two complete sets: with and without the observed-flood outline
# =====================================================================
by_key = {i["key"]: i for i in infos}
MAP_SETS = ([("maps_with_flood_outline", True)] if flood_layer else []) + [("maps_without_flood_outline", False)]
for set_name, mode in MAP_SETS:
    OVERLAY_ON, SET_TAG = mode, set_name
    for S in ((PUB, PRES) if EXPORT_PRESENTATION else (PUB,)):
        folder = os.path.join(out_root, set_name, S["name"])
        os.makedirs(folder, exist_ok=True)
        for i, info in enumerate(infos):
            export(single_layout(info, i, S), folder, f"{i + 1:02d}_{info['key']}", S)
    for cname, cfg in COMBINED_SETS.items():
        chosen = [by_key[k] for k in cfg["keys"] if k in by_key]
        if chosen:
            export(combined_layout(f"{set_name}_{cname}", chosen, PUB, ncols=min(cfg["ncols"], len(chosen)),
                                   map_hmax=cfg["hmax"]),
                   os.path.join(out_root, set_name, PUB["name"]), cname, PUB)

shutil.rmtree(tmp_dir, ignore_errors=True)

print("\nDONE.")
for set_name, _ in MAP_SETS:
    print("  Maps              :", os.path.join(out_root, set_name))
print("  Tables (CSV)      :", tab_dir)
print("  Charts            :", fig_dir)
print("  Rasters           :", rast_dir)
print(f"  Susceptibility: High+Very high = {ar:.1f}% of area, {cap:.1f}% of observed flood, full-extent AUC {auc_full:.3f}")