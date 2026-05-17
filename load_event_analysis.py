"""
PJM DataMiner2 — July 10, 2024 Load Disconnection/Reconnection Event
=====================================================================
Pulls real-time load and transmission constraint data from the PJM
DataMiner2 API and produces an annotated visualization of the
disconnection/reconnection event.

Usage:
    pip install requests pandas matplotlib python-dotenv

    Set your API key via environment variable or pass it directly:
        export PJM_API_KEY="your_key_here"
        python pjm_load_event_analysis.py

    Or edit the API_KEY constant below.
"""

import os
import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY = os.environ.get("PJM_API_KEY", "YOUR_API_KEY_HERE")

BASE_URL = "https://api.pjm.com/api/v1"

# Event window — expand if your event falls outside this range.
# PJM's DataMiner API expects M/D/YYYY HH:MM, not ISO.
EVENT_DATE      = "2024-07-10"
WINDOW_START_ET = "7/9/2024 00:00"    # Eastern Time
WINDOW_END_ET   = "7/12/2024 00:00"

# PJM zone to highlight — set to None to plot RTO-wide instantaneous load
# Examples: "PEPCO", "PSEG", "BGE", "COMED", "PECO", "PPL", "AEP", "DUQUESNE"
ZONE_FILTER = "DOM"   # None = RTO aggregate

# Output file
OUTPUT_PNG = "pjm_july10_2024_event.png"

ET = ZoneInfo("America/New_York")

# ── Helpers ──────────────────────────────────────────────────────────────────

def pjm_get(endpoint: str, params: dict) -> pd.DataFrame:
    """
    Generic DataMiner2 GET wrapper.
    Handles pagination (rowCount / startRow) and returns a flat DataFrame.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": API_KEY,
    }
    rows = []
    params = dict(params)
    params.setdefault("rowCount", 5000)
    params.setdefault("startRow", 1)

    while True:
        resp = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        # DataMiner2 wraps results in a list or a dict with an 'items' key
        if isinstance(payload, list):
            batch = payload
        elif isinstance(payload, dict):
            # Some feeds wrap in {"items": [...], "totalRows": N}
            batch = payload.get("items", payload.get("data", []))
        else:
            batch = []

        rows.extend(batch)

        # Pagination: if we got a full page, ask for the next
        if isinstance(payload, dict) and len(batch) == params["rowCount"]:
            params["startRow"] += params["rowCount"]
        else:
            break

    return pd.DataFrame(rows)


# ── Data Fetchers ─────────────────────────────────────────────────────────────

def fetch_instantaneous_load() -> pd.DataFrame:
    """
    Feed: hrl_load_metered  (hourly metered load by zone, MW — retained indefinitely)
    Columns of interest: datetime_beginning_ept, mw, zone, load_area
    """
    print("Fetching hourly metered load data …")
    params = {
        "startRow":   1,
        "rowCount":   5000,
        "datetime_beginning_ept": f"{WINDOW_START_ET} to {WINDOW_END_ET}",
    }
    if ZONE_FILTER:
        params["zone"] = ZONE_FILTER

    df = pjm_get("hrl_load_metered", params)

    if df.empty:
        raise ValueError("hrl_load_metered returned no data. Check your API key and date range.")

    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse timestamp (prefer EPT for plotting). PJM returns naive strings
    # already in Eastern Time — localize so matplotlib formats correctly.
    ts_col = "datetime_beginning_ept" if "datetime_beginning_ept" in df.columns else next(
        (c for c in df.columns if "datetime" in c), None
    )
    if ts_col is None:
        raise KeyError(f"No timestamp column found. Available: {list(df.columns)}")
    df["ts"] = pd.to_datetime(df[ts_col]).dt.tz_localize(ET, ambiguous="NaT", nonexistent="NaT")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    if "mw" not in df.columns:
        raise KeyError(f"No 'mw' column found. Available: {list(df.columns)}")
    df["load_mw"] = pd.to_numeric(df["mw"], errors="coerce")

    # Zone filtering (API already filtered if ZONE_FILTER was set, but keep a guard)
    zone_col = "zone" if "zone" in df.columns else ("load_area" if "load_area" in df.columns else None)
    if ZONE_FILTER and zone_col:
        zone_df = df[df[zone_col].str.upper() == ZONE_FILTER.upper()].copy()
        if zone_df.empty:
            available = df[zone_col].unique().tolist()
            print(f"  WARNING: zone '{ZONE_FILTER}' not found. Available: {available}")
            print("  Falling back to RTO aggregate.")
            zone_df = _aggregate_rto(df)
    else:
        zone_df = _aggregate_rto(df)

    return zone_df[["ts", "load_mw"]].dropna()


def _aggregate_rto(df: pd.DataFrame) -> pd.DataFrame:
    """Sum all zones per timestamp to get RTO instantaneous load."""
    return df.groupby("ts", as_index=False)["load_mw"].sum()


# ── Event Detection ───────────────────────────────────────────────────────────

def detect_event(load_df: pd.DataFrame,
                 drop_threshold_mw: float = 1500.0,
                 recovery_fraction: float = 0.5,
                 recovery_search_hours: int = 12) -> tuple:
    """
    Find the largest sudden load drop and (best-effort) its recovery.
    Returns (disconnect_ts, reconnect_ts). reconnect_ts may be None.

    Parameters
    ----------
    drop_threshold_mw     : minimum step-over-step MW drop to flag
    recovery_fraction     : recovery counts when a later step rises by at
                            least this fraction of the original drop magnitude
    recovery_search_hours : how far past the disconnect to look for recovery
    """
    df = load_df.sort_values("ts").reset_index(drop=True).copy()
    df["delta"] = df["load_mw"].diff()

    drops = df[df["delta"] <= -drop_threshold_mw]
    if drops.empty:
        return None, None

    # Pick the deepest single-step drop
    idx = drops["delta"].idxmin()
    disconnect_ts = df.loc[idx, "ts"]
    drop_magnitude = -df.loc[idx, "delta"]

    # Look ahead for a comparable rebound
    horizon_end = disconnect_ts + pd.Timedelta(hours=recovery_search_hours)
    forward = df[(df["ts"] > disconnect_ts) & (df["ts"] <= horizon_end)]
    rebounds = forward[forward["delta"] >= drop_magnitude * recovery_fraction]
    reconnect_ts = rebounds.iloc[0]["ts"] if not rebounds.empty else None

    return disconnect_ts, reconnect_ts


# ── Plotting ──────────────────────────────────────────────────────────────────

def build_figure(load_df: pd.DataFrame,
                 disconnect_ts,
                 reconnect_ts) -> plt.Figure:

    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")
    ax1.set_facecolor("#0f1117")
    ax1.tick_params(colors="#cccccc", labelsize=9)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333344")

    date_fmt = mdates.DateFormatter("%m/%d %H:%M", tz=ET)

    ax1.plot(load_df["ts"], load_df["load_mw"],
             color="#4fc3f7", linewidth=1.4, label="Metered Load (MW)")

    # Rolling baseline (3-hour window matches detect_event for hourly data)
    load_idx = load_df.set_index("ts")["load_mw"]
    baseline = load_idx.rolling("3h", center=True, min_periods=3).mean()
    ax1.plot(baseline.index, baseline.values,
             color="#ffffff", linewidth=0.8, linestyle="--",
             alpha=0.4, label="3-hour rolling baseline")

    ymax = load_df["load_mw"].max()
    if disconnect_ts is not None:
        ax1.axvline(disconnect_ts, color="#ff6b6b", linewidth=1.5,
                    linestyle="--", alpha=0.85)
        disc_label = disconnect_ts.strftime("%m/%d %H:%M ET")
        ax1.text(disconnect_ts, ymax * 0.97, f"↓ Drop\n{disc_label}",
                 color="#ff9999", fontsize=8, ha="left", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#1a0a0a", ec="#ff6b6b", alpha=0.8))

    if disconnect_ts is not None and reconnect_ts is not None:
        ax1.axvspan(disconnect_ts, reconnect_ts,
                    color="#ff6b6b", alpha=0.18, label="Event window")
        ax1.axvline(reconnect_ts, color="#69db7c", linewidth=1.5,
                    linestyle="--", alpha=0.85)
        recon_label = reconnect_ts.strftime("%m/%d %H:%M ET")
        ax1.text(reconnect_ts, ymax * 0.97, f"↑ Recovery\n{recon_label}",
                 color="#a9e34b", fontsize=8, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#0a1a0a", ec="#69db7c", alpha=0.8))

    title_zone = ZONE_FILTER if ZONE_FILTER else "RTO Aggregate"
    ax1.set_ylabel("Load (MW)", color="#cccccc", fontsize=10)
    ax1.set_xlabel("Time (Eastern)", color="#cccccc", fontsize=10)
    ax1.set_title(
        f"PJM Load Drop Event  —  July 10 2024  ({title_zone})",
        color="white", fontsize=12, pad=12, fontweight="bold"
    )
    ax1.legend(loc="lower right", fontsize=8,
               facecolor="#1c1c2e", edgecolor="#444466", labelcolor="#cccccc")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.xaxis.set_major_formatter(date_fmt)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6, tz=ET))
    ax1.grid(axis="both", color="#1e1e2e", linewidth=0.5)

    if disconnect_ts is not None:
        if reconnect_ts is not None:
            dur_min = (reconnect_ts - disconnect_ts).total_seconds() / 60
            footer = f"Event duration: {dur_min:.0f} min  |  Source: PJM DataMiner2"
        else:
            footer = "No recovery detected within search horizon.  |  Source: PJM DataMiner2"
    else:
        footer = "No significant drop auto-detected.  Adjust threshold or annotate manually.  |  Source: PJM DataMiner2"
    fig.text(0.99, 0.01, footer, ha="right", va="bottom", color="#555577", fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        sys.exit(
            "ERROR: Set PJM_API_KEY environment variable or edit the API_KEY constant.\n"
            "  export PJM_API_KEY='your_key_here'\n"
            "  python pjm_load_event_analysis.py"
        )

    # 1. Fetch data
    load_df = fetch_instantaneous_load()
    print(f"  Loaded {len(load_df)} load rows.")

    # 2. Detect event
    disconnect_ts, reconnect_ts = detect_event(load_df)
    if disconnect_ts is not None:
        disc_str = disconnect_ts.strftime("%m/%d %H:%M ET")
        if reconnect_ts is not None:
            recon_str = reconnect_ts.strftime("%m/%d %H:%M ET")
            dur_min   = (reconnect_ts - disconnect_ts).total_seconds() / 60
            print(f"\n  ✓ Event detected: drop={disc_str}  recovery={recon_str}  duration={dur_min:.0f} min")
        else:
            print(f"\n  ✓ Drop detected at {disc_str} (no matching recovery within search horizon)")
    else:
        print("\n  ⚠  No significant load drop auto-detected.")
        print("     Lower drop_threshold_mw or set disconnect_ts / reconnect_ts manually.")

    # 3. Plot
    fig = build_figure(load_df, disconnect_ts, reconnect_ts)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Chart saved → {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()