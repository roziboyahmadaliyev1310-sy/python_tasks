"""
climate_pipeline.py
End-to-end pipeline: live ingestion (OpenWeatherMap, NOAA small pulls), historical CSVs (GHCN-style),
unit normalization, timezone alignment, anomaly detection, seasonal decomposition, trend calc,
and visualizations: decadal heatmap + continent correlation matrix.

Configure by environment variables:
  OWM_KEY  - OpenWeatherMap API key
  NOAA_TOKEN - NOAA CDO token (optional; bulk GHCN downloads recommended for full archives)

Example usage:
  python climate_pipeline.py --historical data/ghcn_sample.csv --stations data/stations.csv
"""

import os
import sys
import math
import json
import argparse
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# optional imports with fallbacks
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except Exception:
    HAS_STL = False

try:
    import pycountry
    import pycountry_convert as pc
    HAS_PYCOUNTRY = True
except Exception:
    HAS_PYCOUNTRY = False

# KD-tree for spatial nearest (no geopandas required)
from scipy.spatial import cKDTree


# -----------------------
# 1) Small helpers: API fetchers
# -----------------------
OWM_ONECALL_URL = "https://api.openweathermap.org/data/2.5/onecall"
OWM_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
NOAA_CDO_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"

def fetch_owm_current(lat: float, lon: float, api_key: Optional[str] = None, units: str = "metric", timeout: int = 15) -> Dict[str,Any]:
    """Fetch current weather (simple) and return normalized record."""
    api_key = api_key or os.getenv("OWM_KEY")
    if not api_key:
        raise RuntimeError("OpenWeatherMap API key required in OWM_KEY or parameter")
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": units}
    r = requests.get(OWM_CURRENT_URL, params=params, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    ts = datetime.fromtimestamp(int(j.get("dt", datetime.now(timezone.utc).timestamp())), tz=timezone.utc)
    temp = j.get("main",{}).get("temp")
    if temp is None:
        raise RuntimeError("No temp in OWM response")
    temp_c = temp if units=="metric" else (temp - 32.0) * 5.0/9.0
    precip_mm = 0.0
    rain = j.get("rain") or {}
    snow = j.get("snow") or {}
    if "1h" in rain:
        precip_mm = float(rain["1h"])
    elif "3h" in rain:
        precip_mm = float(rain["3h"]) / 3.0
    elif "1h" in snow:
        precip_mm = float(snow["1h"])
    elif "3h" in snow:
        precip_mm = float(snow["3h"]) / 3.0

    return {"timestamp_utc": ts, "lat": float(j["coord"]["lat"]), "lon": float(j["coord"]["lon"]), 
            "temp_C": float(temp_c), "precip_mm": float(precip_mm), "raw": j}


def fetch_noaa_cdo(endpoint: str, token: Optional[str], params: Optional[Dict]=None, timeout:int=20) -> Dict:
    """Simple NOAA CDO small-query wrapper. Use token from NOAA_TOKEN.
       For large historical pulls use bulk GHNC archives instead."""
    if token is None:
        raise RuntimeError("NOAA token required (NOAA_TOKEN env or parameter) for CDO API calls")
    headers = {"token": token}
    url = f"{NOAA_CDO_BASE}/{endpoint}"
    r = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


# -----------------------
# 2) Historical CSV loader & normalization
# -----------------------
def load_historical_csvs(paths: List[str]) -> pd.DataFrame:
    """Load one or more GHCN-style CSVs and concat. Expect columns:
       station_id, date, element (TMAX/TMIN/PRCP), value, lat, lon, country (optional)"""
    dfs = []
    for p in paths:
        dfs.append(pd.read_csv(p, parse_dates=["date"]))
    df = pd.concat(dfs, ignore_index=True)
    return df

def pivot_and_clean_units(ghcn_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot GHCN-like tall dataframe to wide per (station_id, date) with canonical columns:
       tmax_C, tmin_C, tmean_C, prcp_mm
       Handles GHCN tenths-of-degC and PRCP in 0.1mm? (assume mm). Adjust if needed.
    """
    df = ghcn_df.copy()
    # convert tenths C -> C for temperature elements
    temp_mask = df['element'].isin(['TMAX','TMIN','TAVG','TMEAN','T'])
    if temp_mask.any():
        # only scale if values look like tenths (optional heuristic)
        # We'll assume GHCN format (tenths °C)
        df.loc[temp_mask, 'value'] = df.loc[temp_mask, 'value'] / 10.0

    wide = df.pivot_table(index=["station_id","date"], columns="element", values="value", aggfunc='first').reset_index()
    # rename known columns
    if 'TMAX' in wide.columns:
        wide = wide.rename(columns={'TMAX':'tmax_C'})
    if 'TMIN' in wide.columns:
        wide = wide.rename(columns={'TMIN':'tmin_C'})
    if 'TAVG' in wide.columns:
        wide = wide.rename(columns={'TAVG':'tmean_C'})
    if 'PRCP' in wide.columns:
        wide = wide.rename(columns={'PRCP':'prcp_mm'})
    # compute mean if missing
    if 'tmean_C' not in wide.columns and {'tmax_C','tmin_C'}.issubset(wide.columns):
        wide['tmean_C'] = wide[['tmax_C','tmin_C']].mean(axis=1)
    return wide

# -----------------------
# 3) Spatial join (KD-tree) to map point -> nearest station (no GeoPandas required)
# -----------------------
def build_station_tree(stations_df: pd.DataFrame) -> cKDTree:
    """stations_df must have 'lat' and 'lon' columns."""
    pts = np.vstack([stations_df['lat'].values, stations_df['lon'].values]).T
    return cKDTree(pts)

def find_nearest_station(tree: cKDTree, stations_df: pd.DataFrame, lat: float, lon: float):
    dist, idx = tree.query([lat, lon], k=1)
    return stations_df.iloc[int(idx)]

# -----------------------
# 4) Timezone alignment
# -----------------------
def align_to_utc(df: pd.DataFrame, ts_col: str='date', tz_col: Optional[str]=None, assume_naive_tz='UTC') -> pd.DataFrame:
    """Convert ts_col to timezone-aware UTC. If tz_col provided uses per-row IANA tz strings."""
    if ts_col not in df.columns:
        raise KeyError(f"{ts_col} not in DataFrame")
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    if tz_col and tz_col in df.columns:
        out = []
        for ts, tz in zip(df[ts_col], df[tz_col]):
            if pd.isna(ts):
                out.append(pd.NaT); continue
            try:
                if pd.isna(tz):
                    # if ts naive assume assume_naive_tz, else convert
                    if ts.tzinfo is None:
                        out.append(ts.tz_localize(assume_naive_tz).tz_convert('UTC'))
                    else:
                        out.append(ts.tz_convert('UTC'))
                else:
                    if ts.tzinfo is None:
                        out.append(ts.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward').tz_convert('UTC'))
                    else:
                        out.append(ts.tz_convert('UTC'))
            except Exception:
                # fallback: try assume_naive_tz
                try:
                    if ts.tzinfo is None:
                        out.append(ts.tz_localize(assume_naive_tz).tz_convert('UTC'))
                    else:
                        out.append(ts.tz_convert('UTC'))
                except Exception:
                    out.append(pd.NaT)
        df[ts_col] = pd.Series(out, index=df.index)
    else:
        # vectorized: localize naive to assume_naive_tz, convert tz-aware to UTC
        if df[ts_col].dt.tz is None:
            df[ts_col] = df[ts_col].dt.tz_localize(assume_naive_tz)
        df[ts_col] = df[ts_col].dt.tz_convert('UTC')
    return df

# -----------------------
# 5) Anomaly detection
# -----------------------
def detect_anomalies_zscore(series: pd.Series, z_thresh: float=3.5) -> pd.Series:
    vals = series.astype(float).values
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    if mad == 0 or math.isnan(mad):
        zs = (vals - np.nanmean(vals)) / np.nanstd(vals)
    else:
        zs = 0.6745 * (vals - med) / mad
    return pd.Series(np.abs(zs) > z_thresh, index=series.index)

def detect_anomalies_iforest(df: pd.DataFrame, features: List[str], contamination:float=0.01) -> pd.Series:
    X = df[features].fillna(method='ffill').fillna(method='bfill').values
    clf = IsolationForest(contamination=contamination, random_state=0)
    preds = clf.fit_predict(X)
    return pd.Series(preds == -1, index=df.index)

# -----------------------
# 6) Seasonal decomposition (STL preferred, rolling fallback)
# -----------------------
def seasonal_decompose_series(series: pd.Series, period:int=12):
    """Return tuple (trend, seasonal, resid). Uses STL if available, else rolling fallback."""
    s = series.dropna()
    if len(s) < period*2:
        # Too short to decompose reliably
        return (pd.Series(index=series.index, dtype=float),
                pd.Series(index=series.index, dtype=float),
                pd.Series(index=series.index, dtype=float))
    if HAS_STL:
        stl = STL(s, period=period, robust=True)
        res = stl.fit()
        # align to full index (NaN where original missing)
        trend = res.trend.reindex(series.index)
        seasonal = res.seasonal.reindex(series.index)
        resid = res.resid.reindex(series.index)
        return trend, seasonal, resid
    else:
        # fallback: monthly climatology + rolling trend
        monthly_clim = series.groupby(series.index.month).transform('mean')
        trend = series.rolling(window=period, center=True, min_periods=period//2).mean()
        seasonal = monthly_clim
        resid = series - trend - seasonal
        return trend, seasonal, resid

# -----------------------
# 7) Trend calculation (°C/decade)
# -----------------------
def compute_linear_trend_per_series(series: pd.Series):
    s = series.dropna()
    if len(s) < 10:
        return (np.nan, np.nan, np.nan)
    x = s.index.year + (s.index.dayofyear / 365.25)
    y = s.values.astype(float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    slope_dec = slope * 10.0
    return slope_dec, p_value, r_value

# -----------------------
# 8) Continent mapping (pycountry fallback)
# -----------------------
MINIMAL_COUNTRY_TO_CONT = {
    # alpha-2 -> continent name (a minimal subset); expand as needed
    'US':'North America','CA':'North America','MX':'North America',
    'BR':'South America','AR':'South America',
    'GB':'Europe','FR':'Europe','DE':'Europe','IT':'Europe','ES':'Europe',
    'CN':'Asia','JP':'Asia','KR':'Asia','IN':'Asia',
    'AU':'Oceania','NZ':'Oceania','ZA':'Africa','EG':'Africa'
}

def country_to_continent(country_code_or_name: str):
    if country_code_or_name is None or (isinstance(country_code_or_name,float) and math.isnan(country_code_or_name)):
        return None
    s = str(country_code_or_name).strip()
    # if looks like alpha2
    if len(s) == 2 and s.upper() in MINIMAL_COUNTRY_TO_CONT:
        return MINIMAL_COUNTRY_TO_CONT[s.upper()]
    if HAS_PYCOUNTRY:
        try:
            # try alpha2
            c = pycountry.countries.get(alpha_2=s.upper())
            if c is None:
                c = pycountry.countries.search_fuzzy(s)[0]
            alpha2 = c.alpha_2
            code = pc.country_alpha2_to_continent_code(alpha2)
            map_code = {'AF':'Africa','AS':'Asia','EU':'Europe','NA':'North America','OC':'Oceania','SA':'South America','AN':'Antarctica'}
            return map_code.get(code)
        except Exception:
            return None
    else:
        # naive search in minimal map by name
        for k,v in MINIMAL_COUNTRY_TO_CONT.items():
            if s.lower() in ['', v.lower()]:
                return v
        return None

# -----------------------
# 9) Visualizations
# -----------------------
def plot_decadal_heatmap(trends_df: pd.DataFrame, lat_bin_size:int=5, lon_bin_size:int=5, value_col='trend_dec_C_per_decade', cmap='RdBu_r'):
    # require trends_df with 'lat' and 'lon' and trend column
    df = trends_df.dropna(subset=['lat','lon',value_col]).copy()
    df['lat_bin'] = (df['lat']//lat_bin_size)*lat_bin_size
    df['lon_bin'] = (df['lon']//lon_bin_size)*lon_bin_size
    grid = df.groupby(['lat_bin','lon_bin'])[value_col].median().reset_index()
    pivot = grid.pivot(index='lat_bin', columns='lon_bin', values=value_col)
    plt.figure(figsize=(14,7))
    sns.heatmap(pivot.sort_index(ascending=False), center=0, cmap=cmap)
    plt.title('Decadal temperature trend (°C/decade)')
    plt.xlabel('Longitude bin')
    plt.ylabel('Latitude bin')
    plt.tight_layout()
    plt.show()

def plot_continent_correlation(trends_df: pd.DataFrame, feature_cols: List[str]):
    # aggregate per continent then plot correlation of features
    df = trends_df.copy()
    df['continent'] = df['country'].apply(country_to_continent) if 'country' in df.columns else df.get('continent')
    agg = df.groupby('continent')[feature_cols].median().dropna(how='all')
    if agg.shape[1] < 2:
        print("Not enough features to correlate. Need at least 2 features (e.g., temp trend, prcp trend).")
        return
    corr = agg.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap='coolwarm')
    plt.title('Correlation matrix between continents (median aggregated features)')
    plt.tight_layout()
    plt.show()

# -----------------------
# 10) Orchestrator example (small-scale)
# -----------------------
def run_pipeline(historical_paths: List[str], stations_path: str, owm_key: Optional[str]=None, noaa_token: Optional[str]=None):
    print("Loading historical CSV(s)...")
    raw = load_historical_csvs(historical_paths)
    print(f"Loaded {len(raw)} raw records.")

    print("Pivoting and normalizing units...")
    wide = pivot_and_clean_units(raw)
    # ensure station lat/lon present: try merge with stations metadata
    stations = pd.read_csv(stations_path)
    # stations must include columns: station_id, lat, lon, country (optional)
    merged = wide.merge(stations, on='station_id', how='left')

    # monthly aggregation per station
    merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
    merged = merged.dropna(subset=['date'])
    monthly = merged.set_index('date').groupby('station_id').resample('M').mean().reset_index()

    # detect anomalies on tmean_C per station and drop them
    print("Detecting anomalies (zscore) and removing...")
    def drop_anoms(g):
        if 'tmean_C' not in g:
            return g
        mask = detect_anomalies_zscore(g['tmean_C'], z_thresh=3.5)
        return g.loc[~mask]
    monthly_clean = monthly.groupby('station_id', group_keys=False).apply(drop_anoms)

    # compute per-station trend (°C/decade)
    print("Computing trends per station...")
    trends = []
    for sid, g in monthly_clean.groupby('station_id'):
        g = g.set_index('date').sort_index()
        if 'tmean_C' not in g:
            continue
        series = g['tmean_C']
        slope_dec, p, r = compute_linear_trend_per_series(series)
        meta = stations.loc[stations['station_id']==sid].iloc[0].to_dict() if sid in list(stations['station_id']) else {}
        trends.append({**{'station_id':sid,'trend_dec_C_per_decade':slope_dec,'pval':p,'r':r}, **meta})
    trends_df = pd.DataFrame(trends)
    print(f"Computed trends for {len(trends_df)} stations.")

    # heatmap
    plot_decadal_heatmap(trends_df)

    # compute extra features for continent correlation (if prcp trends exist, compute similarly)
    # For demo, we'll compute only temp trend; so correlation needs more features to be useful.
    # If you have prcp trends, include them in feature_cols below.
    feature_cols = ['trend_dec_C_per_decade']  # extend with 'prcp_trend' etc.
    plot_continent_correlation(trends_df, feature_cols)

    return {
        'monthly_clean': monthly_clean,
        'trends_df': trends_df
    }

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--historical', nargs='+', required=True, help='Paths to historical CSV files (GHCN-like)')
    p.add_argument('--stations', required=True, help='Station metadata CSV with station_id, lat, lon, country')
    p.add_argument('--owm_key', default=None, help='OpenWeatherMap API key (env OWM_KEY used if not provided)')
    p.add_argument('--noaa_token', default=None, help='NOAA CDO token (env NOAA_TOKEN used if not provided)')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    owm = args.owm_key or os.getenv('OWM_KEY')
    noaa = args.noaa_token or os.getenv('NOAA_TOKEN')
    out = run_pipeline(args.historical, args.stations, owm, noaa)
    print("Pipeline finished.")





















# """
# dataprocessing.py
# Utilities to ingest OpenWeatherMap current data, convert units, and align timestamps to UTC.
# """

# from datetime import datetime
# import os
# import requests
# from typing import Optional, Dict, Any

# import pandas as pd


# OWM_URL = "https://api.openweathermap.org/data/2.5/weather"


# def fetch_current_by_coord(lat: float, lon: float, api_key: Optional[str] = None,
#                            units: str = "metric", timeout: int = 10) -> Dict[str, Any]:
#     """
#     Fetch current weather from OpenWeatherMap for given coordinates.
#     Returns normalized dict with timestamp (UTC), lat, lon, temp_C, precip_mm.
#     """
#     api_key = api_key or os.getenv("OWM_API_KEY")
#     if not api_key:
#         raise RuntimeError("OWM_API_KEY not provided (env or parameter)")

#     params = {"lat": lat, "lon": lon, "appid": api_key, "units": units}
#     r = requests.get(OWM_URL, params=params, timeout=timeout)
#     r.raise_for_status()
#     data = r.json()

#     # Timestamp from API (unix UTC)
#     ts = datetime.utcfromtimestamp(data.get("dt", int(datetime.utcnow().timestamp())))

#     temp = data.get("main", {}).get("temp")
#     if temp is None:
#         raise ValueError("temperature not found in response")

#     # Normalize temperature to Celsius
#     temp_c = temp if units == "metric" else (temp - 32.0) * 5.0 / 9.0

#     # Estimate precipitation in mm (try rain -> 1h or 3h, then snow)
#     precip_mm = 0.0
#     rain = data.get("rain") or {}
#     snow = data.get("snow") or {}
#     if "1h" in rain:
#         precip_mm = float(rain["1h"])
#     elif "3h" in rain:
#         precip_mm = float(rain["3h"]) / 3.0
#     elif "1h" in snow:
#         precip_mm = float(snow["1h"])
#     elif "3h" in snow:
#         precip_mm = float(snow["3h"]) / 3.0

#     return {
#         "timestamp_utc": ts,
#         "lat": float(data["coord"]["lat"]),
#         "lon": float(data["coord"]["lon"]),
#         "temp_C": float(temp_c),
#         "precip_mm": float(precip_mm),
#         "raw": data
#     }


# # Unit conversion utilities
# def f_to_c(f: float) -> float:
#     return (f - 32.0) * 5.0 / 9.0


# def c_to_f(c: float) -> float:
#     return c * 9.0 / 5.0 + 32.0


# def inches_to_mm(inches: float) -> float:
#     return inches * 25.4


# def mm_to_inches(mm: float) -> float:
#     return mm / 25.4


# # Timezone alignment
# def align_to_utc(df: pd.DataFrame, ts_col: str = "timestamp", tz_col: Optional[str] = None) -> pd.DataFrame:
#     """
#     Align a DataFrame timestamp column to UTC.

#     - If tz_col is provided and contains IANA timezone strings per row, localize each row accordingly.
#     - Otherwise, if timestamps are timezone-aware they are converted to UTC.
#     - If timestamps are naive and no tz_col, they are assumed to be UTC (no conversion).
#     """
#     if ts_col not in df.columns:
#         raise KeyError(f"{ts_col} not found in DataFrame")

#     # Ensure datetime dtype
#     df = df.copy()
#     df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

#     if tz_col and tz_col in df.columns:
#         # Per-row localization (handles different timezones per row)
#         def localize_row(row):
#             ts = row[ts_col]
#             if pd.isna(ts):
#                 return pd.NaT
#             tz = row[tz_col]
#             if pd.isna(tz):
#                 # assume timestamp already UTC or naive -> treat as UTC
#                 return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
#             # localize naive ts to tz, then convert to UTC
#             if ts.tzinfo is None:
#                 return ts.tz_localize(tz).tz_convert("UTC")
#             else:
#                 return ts.tz_convert("UTC")

#         df[ts_col] = df.apply(localize_row, axis=1)
#     else:
#         # No per-row tz info:
#         # If tz-naive, assume UTC; if tz-aware, convert to UTC.
#         if df[ts_col].dt.tz is None:
#             # localize naive datetimes to UTC
#             df[ts_col] = df[ts_col].dt.tz_localize("UTC")
#         else:
#             df[ts_col] = df[ts_col].dt.tz_convert("UTC")

#     return df


# if __name__ == "__main__":
#     # Quick local test: create a small DataFrame and align timestamps
#     sample = pd.DataFrame({
#         "timestamp": ["2025-11-17 12:00:00", "2025-11-17 08:00:00"],
#         "tz": ["Europe/Berlin", "America/New_York"],
#         "value": [1.0, 2.0]
#     })
#     print("Before:")
#     print(sample)
#     aligned = align_to_utc(sample, ts_col="timestamp", tz_col="tz")
#     print("\nAfter align_to_utc:")
#     print(aligned)