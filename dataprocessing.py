
# ...existing code...
import os
import requests
from datetime import datetime

OWM_URL = "https://api.openweathermap.org/data/2.5/weather"

def fetch_current_by_coord(lat, lon, api_key=None, units="metric", timeout=10):
    api_key = api_key or os.getenv("OWM_API_KEY")
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": units}
    r = requests.get(OWM_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return {
        "timestamp_utc": datetime.utcfromtimestamp(data["dt"]),
        "lat": data["coord"]["lat"],
        "lon": data["coord"]["lon"],
        "temp_C": data["main"]["temp"] if units == "metric" else (data["main"]["temp"] - 32) * 5.0/9.0,
        "precip_mm": 0.0
    }
# ...existing code...

# ...existing code...
def f_to_c(f):
    return (f - 32.0) * 5.0 / 9.0

def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0

def inches_to_mm(inches):
    return inches * 25.4

def mm_to_inches(mm):
    return mm / 25.4
# ...existing code...


# ...existing code...
import pandas as pd

def align_to_utc(df, ts_col="timestamp", tz_col=None):
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if tz_col is not None and tz_col in df.columns:
        # per-row localization when different zones per row
        df[ts_col] = ts.where(ts.isna(), ts.dt.tz_localize(None))
        df[ts_col] = df.apply(
            lambda r: pd.to_datetime(r[ts_col]).tz_localize(r[tz_col]).tz_convert("UTC")
            if pd.notna(r[ts_col]) else r[ts_col],
            axis=1
        )
    else:
        # assume naive times are UTC
        df[ts_col] = ts.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    return df
# ...existing code...