# ============================================================
# Solar Flare Precursor Detection — v3 Pipeline
# GOES-15/16 XRS + SDO/HMI SHARP | Hybrid Anomaly + Supervised
# 2012–2024
# ============================================================

# ============================================================
# SETUP
# ============================================================
# !pip install netCDF4 xarray "sunpy[net]" mpl_animators torch xgboost drms

from google.colab import drive
drive.mount('/content/drive')

import os

PROJECT_DIR = '/content/drive/MyDrive/solar-flare-precursors'

folders = [
    f'{PROJECT_DIR}/data/raw',
    f'{PROJECT_DIR}/data/raw/goes15',
    f'{PROJECT_DIR}/data/raw/goes16',
    f'{PROJECT_DIR}/data/processed',
    f'{PROJECT_DIR}/data/features',
    f'{PROJECT_DIR}/data/flare_catalog',
    f'{PROJECT_DIR}/data/sharp',
    f'{PROJECT_DIR}/notebooks',
    f'{PROJECT_DIR}/results',
    f'{PROJECT_DIR}/models',
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

RAW_DIR       = f'{PROJECT_DIR}/data/raw'
PROCESSED_DIR = f'{PROJECT_DIR}/data/processed'
FEATURES_DIR  = f'{PROJECT_DIR}/data/features'
CATALOG_DIR   = f'{PROJECT_DIR}/data/flare_catalog'
SHARP_DIR     = f'{PROJECT_DIR}/data/sharp'
RESULTS_DIR   = f'{PROJECT_DIR}/results'

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import time
from itertools import permutations

from scipy import stats
from scipy.fft import fft, fftfreq

from sunpy.net import Fido, attrs as a, hek
from sunpy.timeseries import TimeSeries

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import xgboost as xgb
import drms


# ============================================================
# SECTION 1: DOWNLOAD GOES XRS DATA (2012–2024)
# ============================================================
# GOES-15 for 2012–2016 (Solar Cycle 24 rising/maximum)
# GOES-16 for 2017–2024 (Solar Cycle 24 decline + Cycle 25 rise)

SATELLITE_CONFIG = [
    {'years': range(2012, 2017), 'sat_num': 15, 'raw_subdir': 'goes15'},
    {'years': range(2017, 2025), 'sat_num': 16, 'raw_subdir': 'goes16'},
]

for config in SATELLITE_CONFIG:
    sat_dir = f'{RAW_DIR}/{config["raw_subdir"]}'
    for year in config['years']:
        year_files = glob.glob(f'{sat_dir}/*{year}*.nc') + \
                     glob.glob(f'{sat_dir}/*{year}*.fits')
        if len(year_files) > 300:
            print(f"  GOES-{config['sat_num']} {year}: Already have {len(year_files)} files — skipping.")
            continue

        print(f"\nDownloading GOES-{config['sat_num']} {year}...")
        try:
            result = Fido.search(
                a.Time(f'{year}-01-01', f'{year}-12-31'),
                a.Instrument.xrs,
                a.goes.SatelliteNumber(config['sat_num']),
                a.Resolution("avg1m")
            )
            files = Fido.fetch(result, path=f'{sat_dir}/{{file}}')
            print(f"  Downloaded {len(files)} files for {year}")
        except Exception as e:
            print(f"  ERROR downloading GOES-{config['sat_num']} {year}: {e}")
            continue

total_files = sum(
    len(glob.glob(f'{RAW_DIR}/{c["raw_subdir"]}/*.nc') +
        glob.glob(f'{RAW_DIR}/{c["raw_subdir"]}/*.fits'))
    for c in SATELLITE_CONFIG
)
print(f"Total raw files: {total_files}")


# ============================================================
# SECTION 2: LOAD AND HARMONIZE GOES-15/16 DATA
# ============================================================
SAVE_PATH = f'{PROCESSED_DIR}/goes_xrs_2012_2024.csv'

def harmonize_goes_data(goes15_dfs, goes16_dfs):
    """
    Harmonize GOES-15 and GOES-16 XRS data into a single DataFrame.
    Ensures consistent column names and handles satellite overlap.
    """
    all_dfs = []

    for year_df in goes15_dfs:
        # GOES-15 columns vary by product version; normalize
        col_map = {}
        for c in year_df.columns:
            cl = c.lower()
            if 'xrsb' in cl or 'b_flux' in cl:
                col_map[c] = 'xrsb'
            elif 'xrsa' in cl or 'a_flux' in cl:
                col_map[c] = 'xrsa'
            elif 'xrsb_quality' in cl or 'b_qual' in cl:
                col_map[c] = 'xrsb_quality'
            elif 'xrsa_quality' in cl or 'a_qual' in cl:
                col_map[c] = 'xrsa_quality'
        year_df = year_df.rename(columns=col_map)
        year_df['satellite'] = 15
        if 'xrsb' in year_df.columns:
            all_dfs.append(year_df[['xrsb', 'xrsa', 'satellite'] +
                                   [c for c in ['xrsb_quality', 'xrsa_quality']
                                    if c in year_df.columns]])

    for year_df in goes16_dfs:
        year_df['satellite'] = 16
        all_dfs.append(year_df[['xrsb', 'xrsa', 'satellite'] +
                               [c for c in ['xrsb_quality', 'xrsa_quality']
                                if c in year_df.columns]])

    combined = pd.concat(all_dfs).sort_index()
    # For any overlap period, prefer GOES-16
    dupes = combined.index.duplicated(keep=False)
    if dupes.any():
        overlap = combined[dupes]
        keep_16 = overlap['satellite'] == 16
        drop_idx = overlap[~keep_16].index
        combined = combined.drop(drop_idx[drop_idx.duplicated(keep=False) &
                                          combined.loc[drop_idx, 'satellite'].eq(15)])
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined


if os.path.exists(SAVE_PATH):
    print(f"Loading processed file from Drive...")
    df = pd.read_csv(SAVE_PATH, parse_dates=['timestamp'], index_col='timestamp')
else:
    goes15_dfs, goes16_dfs = [], []

    for config in SATELLITE_CONFIG:
        sat_dir = f'{RAW_DIR}/{config["raw_subdir"]}'
        files = sorted(glob.glob(f'{sat_dir}/*.nc') + glob.glob(f'{sat_dir}/*.fits'))
        for year in config['years']:
            year_files = [f for f in files if str(year) in os.path.basename(f)]
            if not year_files:
                continue
            try:
                ts = TimeSeries(year_files, concatenate=True)
                year_df = ts.to_dataframe()
                if config['sat_num'] == 15:
                    goes15_dfs.append(year_df)
                else:
                    goes16_dfs.append(year_df)
                print(f"  GOES-{config['sat_num']} {year}: {len(year_df):,} rows")
            except Exception as e:
                print(f"  GOES-{config['sat_num']} {year}: ERROR — {e}")

    df = harmonize_goes_data(goes15_dfs, goes16_dfs)
    df.index.name = 'timestamp'
    df.to_csv(SAVE_PATH)
    print(f"Saved to: {SAVE_PATH}")

print(f"Shape: {df.shape}  |  {df.index.min()} to {df.index.max()}")


# ============================================================
# SECTION 3: DATA CLEANING
# ============================================================
if 'xrsb_quality' in df.columns and 'xrsa_quality' in df.columns:
    bad_quality = (df['xrsb_quality'] != 0) | (df['xrsa_quality'] != 0)
    df.loc[bad_quality, ['xrsb', 'xrsa']] = np.nan

df.loc[df['xrsb'] <= 0, 'xrsb'] = np.nan
df.loc[df['xrsa'] <= 0, 'xrsa'] = np.nan

df['xrsb'] = df['xrsb'].interpolate(method='linear', limit=10)
df['xrsa'] = df['xrsa'].interpolate(method='linear', limit=10)

df['channel_ratio'] = df['xrsa'] / df['xrsb']
df['channel_ratio'] = df['channel_ratio'].replace([np.inf, -np.inf], np.nan)
df['channel_ratio'] = df['channel_ratio'].interpolate(method='linear', limit=10)

df['has_gap'] = df['xrsb'].isna().astype(int)

n_valid = df['xrsb'].notna().sum()
print(f"Valid data: {n_valid:,} / {len(df):,} ({100*n_valid/len(df):.2f}%)")


# ============================================================
# SECTION 4: VISUALIZE FULL TIME SERIES
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

axes[0].semilogy(df.index, df['xrsb'], linewidth=0.15, color='navy', alpha=0.6)
axes[0].axhline(1e-6, color='green',  linestyle='--', alpha=0.4, label='C')
axes[0].axhline(1e-5, color='orange', linestyle='--', alpha=0.4, label='M')
axes[0].axhline(1e-4, color='red',    linestyle='--', alpha=0.4, label='X')
axes[0].set_ylabel('XRS-B (1–8 Å)\nW/m²')
axes[0].set_title('GOES X-ray Flux — Full Dataset (2012–2024)')
axes[0].legend(loc='upper left', fontsize=8)

axes[1].semilogy(df.index, df['xrsa'], linewidth=0.15, color='darkred', alpha=0.6)
axes[1].set_ylabel('XRS-A (0.5–4 Å)\nW/m²')

axes[2].plot(df.index, df['channel_ratio'], linewidth=0.15, color='darkgreen', alpha=0.6)
axes[2].set_ylabel('Channel Ratio\n(A / B)')
axes[2].set_ylim(0, 2)

for split in [pd.Timestamp('2020-01-01'), pd.Timestamp('2022-01-01')]:
    for ax in axes:
        ax.axvline(split, color='red', linestyle='-', linewidth=1.5, alpha=0.8)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/full_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# SECTION 5: DOWNLOAD FLARE CATALOG (M/X CLASS, 2012–2024)
# ============================================================
CATALOG_PATH = f'{CATALOG_DIR}/mx_flares_2012_2024.csv'

if os.path.exists(CATALOG_PATH):
    print("Loading flare catalog from Drive...")
    mx_flares = pd.read_csv(CATALOG_PATH, parse_dates=['start_time', 'peak_time', 'end_time'])
else:
    client = hek.HEKClient()
    all_flares = []

    for year in range(2012, 2025):
        print(f"  {year}...", end=" ")
        try:
            result = client.search(
                a.Time(f'{year}-01-01', f'{year}-12-31'),
                a.hek.EventType('FL'),
                a.hek.FL.GOESCls >= 'M1.0'
            )
            for event in result:
                all_flares.append({
                    'start_time': event['event_starttime'],
                    'peak_time':  event['event_peaktime'],
                    'end_time':   event['event_endtime'],
                    'flare_class': event['fl_goescls'],
                    'source':     event['frm_name'],
                })
            print(f"found {len(result)} events")
        except Exception as e:
            print(f"ERROR — {e}")

    mx_flares = pd.DataFrame(all_flares)
    mx_flares['start_time'] = pd.to_datetime(mx_flares['start_time'].astype(str))
    mx_flares['peak_time']  = pd.to_datetime(mx_flares['peak_time'].astype(str))
    mx_flares['end_time']   = pd.to_datetime(mx_flares['end_time'].astype(str))
    mx_flares = mx_flares.sort_values('start_time').reset_index(drop=True)

    # Remove duplicates (events < 10 minutes apart)
    keep = [True]
    for i in range(1, len(mx_flares)):
        diff = (mx_flares.loc[i, 'start_time'] - mx_flares.loc[i-1, 'start_time']).total_seconds() / 60
        keep.append(diff > 10)
    mx_flares = mx_flares[keep].reset_index(drop=True)
    mx_flares['class_letter'] = mx_flares['flare_class'].str[0]
    mx_flares.to_csv(CATALOG_PATH, index=False)
    print(f"Saved to: {CATALOG_PATH}")

mx_flares['class_letter'] = mx_flares['flare_class'].str[0]
print(f"M-class: {(mx_flares['class_letter']=='M').sum()}  |  X-class: {(mx_flares['class_letter']=='X').sum()}")


# ============================================================
# SECTION 5b: VISUALIZE FLARE DISTRIBUTION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

mx_flares['year'] = mx_flares['start_time'].dt.year
yearly = mx_flares.groupby(['year', 'class_letter']).size().unstack(fill_value=0)
yearly.plot(kind='bar', ax=axes[0], color={'M': 'steelblue', 'X': 'coral'},
            edgecolor='black', linewidth=0.5)
axes[0].set_title('M/X Flares Per Year (2012–2024)')
axes[0].set_ylabel('Count')
axes[0].axvline(7.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Train/Val split')
axes[0].axvline(9.5, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Val/Test split')
axes[0].legend(fontsize=8)
axes[0].tick_params(axis='x', rotation=45)

m_flares = mx_flares[mx_flares['class_letter'] == 'M']
x_flares_only = mx_flares[mx_flares['class_letter'] == 'X']
axes[1].scatter(m_flares['start_time'], [1]*len(m_flares), s=5, alpha=0.4,
                color='steelblue', label=f'M-class ({len(m_flares)})')
axes[1].scatter(x_flares_only['start_time'], [2]*len(x_flares_only), s=20, alpha=0.7,
                color='coral', label=f'X-class ({len(x_flares_only)})')
axes[1].axvline(pd.Timestamp('2020-01-01'), color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].axvline(pd.Timestamp('2022-01-01'), color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].set_yticks([1, 2])
axes[1].set_yticklabels(['M', 'X'])
axes[1].set_title('Flare Timeline')
axes[1].legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/flare_distribution.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# SECTION 6: DOWNLOAD SDO/HMI SHARP PARAMETERS
# ============================================================
SHARP_KEYWORDS = ['T_REC', 'NOAA_AR', 'HARPNUM',
                   'R_VALUE', 'TOTUSJH', 'TOTPOT',
                   'MEANPOT', 'TOTUSJZ', 'USFLUX', 'AREA_ACR']

SHARP_SAVE_PATH = f'{SHARP_DIR}/sharp_aggregated_2012_2024.csv'


def download_sharp_data(sharp_dir, start_year=2012, end_year=2024):
    """Download SDO/HMI SHARP parameters from JSOC via drms, month-by-month."""
    client = drms.Client()
    all_months = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_file = f'{sharp_dir}/sharp_{year}_{month:02d}.csv'
            if os.path.exists(month_file):
                month_df = pd.read_csv(month_file)
                all_months.append(month_df)
                continue

            # Build JSOC query string
            if month == 12:
                end_str = f'{year+1}.01.01_00:00:00_TAI'
            else:
                end_str = f'{year}.{month+1:02d}.01_00:00:00_TAI'
            query = (f'hmi.sharp_cea_720s'
                     f'[{year}.{month:02d}.01_00:00:00_TAI-{end_str}]')

            print(f"  SHARP {year}-{month:02d}...", end=" ")
            try:
                keys = client.query(query, key=', '.join(SHARP_KEYWORDS[1:]),  # T_REC is index
                                    rec_index=True)
                if keys is None or len(keys) == 0:
                    print("no data")
                    continue
                if isinstance(keys, pd.DataFrame):
                    keys.to_csv(month_file, index=True)
                    all_months.append(keys)
                    print(f"{len(keys):,} records")
                else:
                    print("unexpected format")
            except Exception as e:
                print(f"ERROR — {e}")
                continue

    if not all_months:
        return pd.DataFrame()
    return pd.concat(all_months, ignore_index=True)


def aggregate_sharp_per_timestamp(sharp_raw_df, resample_freq='1h'):
    """
    Aggregate SHARP parameters across active regions per timestamp.
    R_VALUE, TOTUSJZ → max (most flare-productive AR dominates)
    TOTUSJH, TOTPOT, USFLUX → sum (disk-integrated proxy)
    MEANPOT → area-weighted mean
    Also: n_active_regions count
    """
    if sharp_raw_df.empty:
        return pd.DataFrame()

    # Parse T_REC timestamps
    if 'T_REC' in sharp_raw_df.columns:
        sharp_raw_df['timestamp'] = pd.to_datetime(
            sharp_raw_df['T_REC'].str.replace('_TAI', '').str.replace('_', ' '),
            errors='coerce'
        )
    elif sharp_raw_df.index.name == 'T_REC' or 'T_REC' in str(sharp_raw_df.index.names):
        sharp_raw_df = sharp_raw_df.reset_index()
        sharp_raw_df['timestamp'] = pd.to_datetime(
            sharp_raw_df['T_REC'].astype(str).str.replace('_TAI', '').str.replace('_', ' '),
            errors='coerce'
        )

    sharp_raw_df = sharp_raw_df.dropna(subset=['timestamp'])

    # Convert SHARP columns to numeric
    sharp_cols = ['R_VALUE', 'TOTUSJH', 'TOTPOT', 'MEANPOT', 'TOTUSJZ', 'USFLUX', 'AREA_ACR']
    for col in sharp_cols:
        if col in sharp_raw_df.columns:
            sharp_raw_df[col] = pd.to_numeric(sharp_raw_df[col], errors='coerce')

    grouped = sharp_raw_df.groupby('timestamp')

    agg = pd.DataFrame(index=grouped.groups.keys())
    agg.index.name = 'timestamp'

    # Max across ARs (flare-productive region dominates)
    for col in ['R_VALUE', 'TOTUSJZ']:
        if col in sharp_raw_df.columns:
            agg[col] = grouped[col].max()

    # Sum across ARs (disk-integrated proxy)
    for col in ['TOTUSJH', 'TOTPOT', 'USFLUX']:
        if col in sharp_raw_df.columns:
            agg[col] = grouped[col].sum()

    # Area-weighted mean for MEANPOT
    if 'MEANPOT' in sharp_raw_df.columns and 'AREA_ACR' in sharp_raw_df.columns:
        def _area_weighted_mean(g):
            valid = g[['MEANPOT', 'AREA_ACR']].dropna()
            if len(valid) == 0 or valid['AREA_ACR'].sum() == 0:
                return np.nan
            return np.average(valid['MEANPOT'], weights=valid['AREA_ACR'])
        agg['MEANPOT'] = grouped.apply(_area_weighted_mean)
    elif 'MEANPOT' in sharp_raw_df.columns:
        agg['MEANPOT'] = grouped['MEANPOT'].mean()

    agg['n_active_regions'] = grouped.size()

    # Resample to target frequency with forward-fill (SHARP is 12-min)
    agg = agg.sort_index()
    agg = agg.resample(resample_freq).ffill(limit=24)  # ffill up to 24h
    return agg


if os.path.exists(SHARP_SAVE_PATH):
    print("Loading aggregated SHARP data from Drive...")
    sharp_df = pd.read_csv(SHARP_SAVE_PATH, parse_dates=['timestamp'], index_col='timestamp')
else:
    print("Downloading SHARP parameters from JSOC (this may take a while)...")
    sharp_raw = download_sharp_data(SHARP_DIR)
    sharp_df = aggregate_sharp_per_timestamp(sharp_raw)
    if not sharp_df.empty:
        sharp_df.to_csv(SHARP_SAVE_PATH)
        print(f"Saved aggregated SHARP: {sharp_df.shape}")
    else:
        print("WARNING: No SHARP data retrieved. Pipeline will continue with XRS-only features.")

print(f"SHARP data shape: {sharp_df.shape if not sharp_df.empty else 'EMPTY'}")


# ============================================================
# SECTION 7: RUNNING BASELINE COMPUTATION
# ============================================================
# 7-day rolling baseline for cycle-phase-aware normalization.
# Excludes the most recent 1 day to prevent flare contamination.

def compute_running_baseline(df, window_days=7, gap_days=1):
    """
    Compute running baseline (median, std of log10 xrsb) over a trailing window.
    Window: [t - window_days, t - gap_days]
    Computed on hourly-resampled data for efficiency, then reindexed to 1-min.
    """
    hourly = df['xrsb'].resample('1h').median()
    log_hourly = np.log10(hourly.clip(lower=1e-10))

    window_h = window_days * 24
    gap_h = gap_days * 24

    baseline_med = log_hourly.rolling(
        window=window_h, min_periods=24
    ).median().shift(gap_h)

    baseline_std = log_hourly.rolling(
        window=window_h, min_periods=24
    ).std().shift(gap_h)

    # Reindex to original 1-min cadence via forward-fill
    baseline = pd.DataFrame({
        'baseline_log_median': baseline_med,
        'baseline_log_std': baseline_std
    })
    baseline = baseline.reindex(df.index, method='ffill')
    return baseline

print("Computing running baseline...")
baseline = compute_running_baseline(df)
df = df.join(baseline)
print(f"Baseline coverage: {df['baseline_log_median'].notna().sum():,} / {len(df):,}")


# ============================================================
# SECTION 8: FEATURE ENGINEERING
# ============================================================

def _permutation_entropy(x, order=3, delay=1):
    n = len(x)
    perms_list = list(permutations(range(order)))
    counts = {p: 0 for p in perms_list}
    for i in range(n - (order - 1) * delay):
        indices = [i + j * delay for j in range(order)]
        pattern = tuple(np.argsort([x[idx] for idx in indices]))
        if pattern in counts:
            counts[pattern] += 1
    total = sum(counts.values())
    if total == 0:
        return 0
    probs = np.array([c / total for c in counts.values() if c > 0])
    return -np.sum(probs * np.log2(probs))


def _sample_entropy(x, m=2, r=None):
    if r is None:
        r = 0.2 * np.std(x)
    if r == 0:
        return 0
    N = len(x)

    def _count_matches(tlen):
        count = 0
        templates = np.array([x[i:i+tlen] for i in range(N - tlen)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0 or A == 0:
        return 0
    return -np.log(A / B)


def compute_window_features(xrsb, xrsa, ratio, baseline_log_med, baseline_log_std):
    """
    Compute features for a single window using RELATIVE features
    to handle solar cycle distribution mismatch.

    Returns dict of ~21 base features, or None if >20% NaN.
    """
    if np.isnan(xrsb).sum() > len(xrsb) * 0.2:
        return None

    xrsb  = pd.Series(xrsb).ffill().bfill().values
    xrsa  = pd.Series(xrsa).ffill().bfill().values
    ratio = pd.Series(ratio).ffill().bfill().values

    log_b = np.log10(np.clip(xrsb, 1e-10, None))

    features = {}

    # --- Relative time-domain (cycle-invariant) ---
    mean_log_b = np.mean(log_b)
    if np.isfinite(baseline_log_med) and np.isfinite(baseline_log_std) and baseline_log_std > 0:
        features['rel_flux_ratio']  = mean_log_b - baseline_log_med
        features['rel_flux_zscore'] = (mean_log_b - baseline_log_med) / baseline_log_std
        features['rel_max_ratio']   = np.max(log_b) - baseline_log_med
        features['rel_min_ratio']   = np.min(log_b) - baseline_log_med
    else:
        features['rel_flux_ratio']  = 0
        features['rel_flux_zscore'] = 0
        features['rel_max_ratio']   = 0
        features['rel_min_ratio']   = 0

    # --- Scale-invariant time-domain (kept from v2) ---
    features['xrsb_std']          = np.std(log_b)
    features['xrsb_skew']         = stats.skew(log_b)
    features['xrsb_kurtosis']     = stats.kurtosis(log_b)
    slope, *_ = stats.linregress(np.arange(len(log_b)), log_b)
    features['xrsb_slope']        = slope
    diffs = np.diff(log_b)
    features['xrsb_max_abs_deriv'] = np.max(np.abs(diffs)) if len(diffs) > 0 else 0

    # --- Temporal derivatives (new, normalized by baseline) ---
    n = len(log_b)
    mid = n // 2
    deriv_half = np.mean(log_b[mid:]) - np.mean(log_b[:mid])
    features['deriv_half_window'] = deriv_half
    # 2-hour derivative (120 points at 1-min cadence)
    if n >= 240:
        features['deriv_2h'] = np.mean(log_b[-120:]) - np.mean(log_b[:120])
    else:
        features['deriv_2h'] = deriv_half

    # --- Frequency-domain (scale-invariant) ---
    detrended = log_b - np.polyval(
        np.polyfit(np.arange(len(log_b)), log_b, 1), np.arange(len(log_b))
    )
    windowed = detrended * np.hanning(len(detrended))
    N = len(windowed)
    yf = np.abs(fft(windowed))[:N//2]
    xf = fftfreq(N, d=60.0)[:N//2]
    yf, xf = yf[1:], xf[1:]
    power = yf ** 2
    power_sum = power.sum()
    power_norm = power / power_sum if power_sum > 0 else power

    features['dominant_freq']    = xf[np.argmax(power)] if len(power) > 0 else 0
    # Normalize dominant_power by total (relative, not absolute)
    features['dominant_power_rel'] = (np.max(power) / power_sum) if power_sum > 0 else 0
    pn_safe = power_norm[power_norm > 0]
    features['spectral_entropy'] = -np.sum(pn_safe * np.log2(pn_safe)) if len(pn_safe) > 0 else 0
    features['spectral_centroid'] = np.sum(xf * power) / power_sum if power_sum > 0 else 0
    cum_power = np.cumsum(power)
    rolloff_idx = np.searchsorted(cum_power, 0.85 * cum_power[-1]) if len(cum_power) > 0 else 0
    features['spectral_rolloff'] = xf[min(rolloff_idx, len(xf)-1)] if len(xf) > 0 else 0

    # --- Complexity (scale-invariant) ---
    features['permutation_entropy'] = _permutation_entropy(log_b, order=3, delay=1)
    step = max(1, len(log_b) // 60)
    features['sample_entropy'] = _sample_entropy(log_b[::step], m=2, r=0.2*np.std(log_b[::step]))

    # --- Multi-channel (scale-invariant) ---
    features['ratio_mean']  = np.mean(ratio)
    features['ratio_std']   = np.std(ratio)
    ratio_slope, *_ = stats.linregress(np.arange(len(ratio)), ratio)
    features['ratio_slope'] = ratio_slope

    return features


def compute_multi_resolution_stats(xrsb_log, baseline_log_med, resolution_min, window_center_idx):
    """
    Compute summary stats at a given resolution around the window center.
    Returns dict with prefixed feature names.
    """
    half = resolution_min // 2
    start = max(0, window_center_idx - half)
    end = min(len(xrsb_log), window_center_idx + half)
    segment = xrsb_log[start:end]

    if len(segment) < resolution_min * 0.5:
        return {}

    prefix = f'res_{resolution_min // 60}h'
    feats = {}
    mean_val = np.mean(segment)
    feats[f'{prefix}_rel_flux'] = mean_val - baseline_log_med if np.isfinite(baseline_log_med) else 0
    feats[f'{prefix}_std'] = np.std(segment)
    if len(segment) > 1:
        slope, *_ = stats.linregress(np.arange(len(segment)), segment)
        feats[f'{prefix}_slope'] = slope
    else:
        feats[f'{prefix}_slope'] = 0

    # Spectral entropy for this resolution
    if len(segment) > 10:
        detrended = segment - np.polyval(np.polyfit(np.arange(len(segment)), segment, 1),
                                          np.arange(len(segment)))
        windowed = detrended * np.hanning(len(detrended))
        yf = np.abs(fft(windowed))[:len(windowed)//2]
        power = yf[1:] ** 2
        ps = power.sum()
        if ps > 0:
            pn = power / ps
            pn_safe = pn[pn > 0]
            feats[f'{prefix}_spectral_entropy'] = -np.sum(pn_safe * np.log2(pn_safe))
        else:
            feats[f'{prefix}_spectral_entropy'] = 0
    else:
        feats[f'{prefix}_spectral_entropy'] = 0

    return feats


def compute_flare_history_feature(window_end_time, flare_times, lambda_decay=0.1):
    """
    Exponential-decay-weighted flare history.
    flare_times: array of flare start timestamps (sorted ascending).
    """
    features = {}
    if len(flare_times) == 0:
        features['flare_decay_score'] = 0.0
        features['hours_since_last_flare'] = 720.0  # 30 days cap
        features['flare_count_24h'] = 0
        features['flare_count_72h'] = 0
        return features

    # Only consider flares BEFORE the window end
    prior = flare_times[flare_times < window_end_time]
    if len(prior) == 0:
        features['flare_decay_score'] = 0.0
        features['hours_since_last_flare'] = 720.0
        features['flare_count_24h'] = 0
        features['flare_count_72h'] = 0
        return features

    hours_since = (window_end_time - prior).total_seconds() / 3600.0
    features['flare_decay_score'] = np.sum(np.exp(-lambda_decay * hours_since))
    features['hours_since_last_flare'] = min(hours_since.min(), 720.0)
    features['flare_count_24h'] = int((hours_since <= 24).sum())
    features['flare_count_72h'] = int((hours_since <= 72).sum())
    return features


# ============================================================
# SECTION 9: EXTRACT FEATURES ACROSS FULL DATASET
# ============================================================
FEATURES_PATH = f'{FEATURES_DIR}/v3_features.csv'

if os.path.exists(FEATURES_PATH):
    print("Loading v3 feature matrix from Drive...")
    feature_df = pd.read_csv(FEATURES_PATH, parse_dates=['window_start', 'window_end'])
else:
    window_size = 360   # 6 hours (base resolution)
    stride      = 60    # 1 hour

    n_windows = (len(df) - window_size) // stride + 1
    print(f"Processing {n_windows:,} windows (this takes 15-30 min)...")

    flare_times = pd.to_datetime(mx_flares['start_time']).sort_values().values

    all_features = []
    skipped = 0
    start_time = time.time()
    ten_pct = max(1, n_windows // 10)

    for i in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[i:i + window_size]
        w_start = window.index[0]
        w_end   = window.index[-1]

        # Running baseline at window start
        bl_med = window['baseline_log_median'].iloc[0]
        bl_std = window['baseline_log_std'].iloc[0]

        # Base 6h window features (relative)
        feats = compute_window_features(
            window['xrsb'].values, window['xrsa'].values,
            window['channel_ratio'].values,
            bl_med, bl_std
        )
        if feats is None:
            skipped += 1
            n_done = len(all_features) + skipped
            if n_done % ten_pct == 0 and n_done > 0:
                elapsed = time.time() - start_time
                eta = (n_windows - n_done) / (n_done / elapsed) / 60
                print(f"  {100*n_done/n_windows:5.1f}% | ~{eta:.1f} min remaining")
            continue

        # Multi-resolution features (1h, 3h, 12h, 24h around window center)
        center_idx = i + window_size // 2
        log_b_full = np.log10(np.clip(df['xrsb'].values, 1e-10, None))
        for res_min in [60, 180, 720, 1440]:
            mr_feats = compute_multi_resolution_stats(log_b_full, bl_med, res_min, center_idx)
            feats.update(mr_feats)

        # Flare history features
        fh_feats = compute_flare_history_feature(
            pd.Timestamp(w_end), pd.to_datetime(flare_times)
        )
        feats.update(fh_feats)

        feats['window_start'] = w_start
        feats['window_end']   = w_end
        all_features.append(feats)

        n_done = len(all_features) + skipped
        if n_done % ten_pct == 0 and n_done > 0:
            elapsed = time.time() - start_time
            eta = (n_windows - n_done) / (n_done / elapsed) / 60
            print(f"  {100*n_done/n_windows:5.1f}% | ~{eta:.1f} min remaining")

    feature_df = pd.DataFrame(all_features)
    feature_df.to_csv(FEATURES_PATH, index=False)
    print(f"Done in {(time.time()-start_time)/60:.1f} min  |  skipped: {skipped:,}")

feature_cols = [c for c in feature_df.columns if c not in ['window_start', 'window_end']]
print(f"Feature matrix: {feature_df.shape}  |  {len(feature_cols)} features")


# ============================================================
# SECTION 9b: CROSS-WINDOW DELTA FEATURES
# ============================================================
feature_df = feature_df.sort_values('window_start').reset_index(drop=True)

for lag_h in [1, 2, 6, 12, 24]:
    lag_rows = lag_h  # stride = 1h
    feature_df[f'delta_rel_ratio_{lag_h}h'] = (
        feature_df['rel_flux_ratio'] - feature_df['rel_flux_ratio'].shift(lag_rows)
    )
    feature_df[f'delta_std_{lag_h}h'] = (
        feature_df['xrsb_std'] - feature_df['xrsb_std'].shift(lag_rows)
    )

# 24-hour background trend on relative feature
feature_df['background_trend_24h'] = (
    feature_df['rel_flux_ratio']
    .rolling(24, min_periods=6)
    .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
)

delta_cols = [c for c in feature_df.columns
              if c.startswith('delta_') or c == 'background_trend_24h']
feature_cols = [c for c in feature_df.columns if c not in ['window_start', 'window_end']]
print(f"After deltas: {len(feature_cols)} total features")


# ============================================================
# SECTION 9c: MERGE SHARP FEATURES
# ============================================================

def merge_sharp_features(feature_df, sharp_df):
    """Merge SHARP parameters into feature matrix via nearest-time join."""
    if sharp_df.empty:
        print("SHARP data not available — skipping SHARP feature merge.")
        return feature_df

    feature_df['_merge_time'] = pd.to_datetime(feature_df['window_start'])
    sharp_merge = sharp_df.reset_index()
    sharp_merge['timestamp'] = pd.to_datetime(sharp_merge['timestamp'])
    sharp_merge = sharp_merge.sort_values('timestamp')
    feature_df = feature_df.sort_values('_merge_time')

    merged = pd.merge_asof(
        feature_df, sharp_merge,
        left_on='_merge_time', right_on='timestamp',
        tolerance=pd.Timedelta('2h'), direction='nearest'
    )
    merged = merged.drop(columns=['_merge_time', 'timestamp'], errors='ignore')

    # Derived SHARP features
    if 'TOTUSJH' in merged.columns and 'USFLUX' in merged.columns:
        merged['sharp_energy_proxy'] = merged['TOTUSJH'] * merged['USFLUX']
    if 'R_VALUE' in merged.columns and 'n_active_regions' in merged.columns:
        merged['sharp_complexity'] = merged['R_VALUE'] * merged['n_active_regions']
    if 'USFLUX' in merged.columns:
        merged['delta_usflux_6h'] = merged['USFLUX'] - merged['USFLUX'].shift(6)

    return merged


feature_df = merge_sharp_features(feature_df, sharp_df)
feature_cols = [c for c in feature_df.columns if c not in ['window_start', 'window_end']]
print(f"After SHARP merge: {len(feature_cols)} total features")


# ============================================================
# SECTION 10: FEATURE INSPECTION & CLEANUP
# ============================================================
feature_df['window_start'] = pd.to_datetime(feature_df['window_start'])
feature_df['window_end']   = pd.to_datetime(feature_df['window_end'])

n_inf = np.isinf(feature_df[feature_cols].values).sum()
if n_inf > 0:
    print(f"Replacing {n_inf} Inf values with NaN")
    feature_df[feature_cols] = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan)

print(feature_df[feature_cols].describe().round(4).to_string())


# ============================================================
# SECTION 11: TRAIN/VAL/TEST SPLIT WITH FLARE EXCLUSION
# ============================================================
# Train: 2012–2019 (solar cycle 24 max + min + cycle 25 start)
# Val:   2020–2021 (solar max — threshold tuning)
# Test:  2022–2024 (solar max — final evaluation)

TRAIN_END  = pd.Timestamp('2020-01-01')
VAL_END    = pd.Timestamp('2022-01-01')

train_mask = feature_df['window_end'] < TRAIN_END
val_mask   = (feature_df['window_start'] >= TRAIN_END) & (feature_df['window_end'] < VAL_END)
test_mask  = feature_df['window_start'] >= VAL_END

# --- Flare exclusion for anomaly detectors (quiet-sun training) ---
train_flares     = mx_flares[mx_flares['start_time'] < TRAIN_END]
exclusion_mask   = pd.Series(False, index=feature_df.index)
exclusion_hours  = 24

for _, flare in train_flares.iterrows():
    flare_start = pd.Timestamp(flare['start_time'])
    flare_end   = pd.Timestamp(flare['end_time']) if pd.notna(flare['end_time']) \
                  else flare_start + pd.Timedelta(hours=1)
    excl_start  = flare_start - pd.Timedelta(hours=exclusion_hours)
    overlap = (
        (feature_df['window_end']   > excl_start) &
        (feature_df['window_start'] < flare_end)
    )
    exclusion_mask = exclusion_mask | overlap

train_clean = train_mask & ~exclusion_mask  # For anomaly detectors

# --- Window labels for supervised classifier ---
def label_windows(feature_df_subset, flare_catalog, pre_flare_hours=24):
    """Create binary labels + distance-decay sample weights for supervised training."""
    starts = feature_df_subset['window_start'].values
    ends   = feature_df_subset['window_end'].values
    labels = np.zeros(len(feature_df_subset), dtype=int)
    weights = np.ones(len(feature_df_subset), dtype=float)

    for _, flare in flare_catalog.iterrows():
        flare_start = pd.Timestamp(flare['start_time'])
        pre_start   = flare_start - pd.Timedelta(hours=pre_flare_hours)
        in_zone = (
            (starts >= np.datetime64(pre_start)) &
            (ends   <= np.datetime64(flare_start))
        )
        labels[in_zone] = 1

        # Distance-decay weights: closer to flare = higher weight
        for idx in np.where(in_zone)[0]:
            w_end = pd.Timestamp(ends[idx])
            hours_before = (flare_start - w_end).total_seconds() / 3600
            if hours_before <= 6:
                weights[idx] = 1.0
            elif hours_before <= 12:
                weights[idx] = max(weights[idx], 0.5)
            else:
                weights[idx] = max(weights[idx], 0.25)

    return labels, weights


print(f"Training windows (clean): {train_clean.sum():,}")
print(f"Training windows (all):   {train_mask.sum():,}")
print(f"Validation windows:       {val_mask.sum():,}")
print(f"Test windows:             {test_mask.sum():,}")

# --- Prepare feature matrices ---
X_train_clean = feature_df.loc[train_clean, feature_cols].values
X_train_all   = feature_df.loc[train_mask,  feature_cols].values
X_val         = feature_df.loc[val_mask,    feature_cols].values
X_test        = feature_df.loc[test_mask,   feature_cols].values

# Labels for supervised training (all training data, labeled)
y_train_all, w_train_all = label_windows(
    feature_df.loc[train_mask], train_flares, pre_flare_hours=24
)

val_flares = mx_flares[
    (mx_flares['start_time'] >= TRAIN_END) & (mx_flares['start_time'] < VAL_END)
]
y_val, w_val = label_windows(
    feature_df.loc[val_mask], val_flares, pre_flare_hours=24
)

# Impute & scale (fit on clean training data only — no leakage)
imputer = SimpleImputer(strategy='median')
X_train_clean = imputer.fit_transform(X_train_clean)
X_train_all   = imputer.transform(X_train_all)
X_val         = imputer.transform(X_val)
X_test        = imputer.transform(X_test)

scaler = StandardScaler()
X_train_clean_scaled = scaler.fit_transform(X_train_clean)
X_train_all_scaled   = scaler.transform(X_train_all)
X_val_scaled         = scaler.transform(X_val)
X_test_scaled        = scaler.transform(X_test)

# Safety net
for arr in [X_train_clean_scaled, X_train_all_scaled, X_val_scaled, X_test_scaled]:
    arr[:] = np.nan_to_num(arr, nan=0, posinf=0, neginf=0)

print(f"Pre-flare windows in train: {y_train_all.sum():,} / {len(y_train_all):,}")
print(f"Pre-flare windows in val:   {y_val.sum():,} / {len(y_val):,}")


# ============================================================
# SECTION 12: TRAIN ANOMALY DETECTORS
# ============================================================

# Isolation Forest
print("Training Isolation Forest...", end=" ")
t0 = time.time()
iso_forest = IsolationForest(n_estimators=200, contamination='auto', random_state=42, n_jobs=-1)
iso_forest.fit(X_train_clean_scaled)
iso_scores_train = -iso_forest.score_samples(X_train_all_scaled)
iso_scores_val   = -iso_forest.score_samples(X_val_scaled)
iso_scores_test  = -iso_forest.score_samples(X_test_scaled)
print(f"done ({time.time()-t0:.1f}s)")

# Local Outlier Factor
print("Training Local Outlier Factor...", end=" ")
t0 = time.time()
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto', n_jobs=-1)
lof.fit(X_train_clean_scaled)
lof_scores_train = -lof.score_samples(X_train_all_scaled)
lof_scores_val   = -lof.score_samples(X_val_scaled)
lof_scores_test  = -lof.score_samples(X_test_scaled)
print(f"done ({time.time()-t0:.1f}s)")

# One-Class SVM
print("Training One-Class SVM...", end=" ")
t0 = time.time()
ocsvm = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
ocsvm.fit(X_train_clean_scaled)
ocsvm_scores_train = -ocsvm.score_samples(X_train_all_scaled)
ocsvm_scores_val   = -ocsvm.score_samples(X_val_scaled)
ocsvm_scores_test  = -ocsvm.score_samples(X_test_scaled)
print(f"done ({time.time()-t0:.1f}s)")


# ============================================================
# SECTION 12b: LSTM AUTOENCODER (48-hour lookback)
# ============================================================

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        repeated = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _   = self.decoder(repeated)
        return out


def make_sequences(X_scaled, seq_len=48):
    """Slide a window of seq_len rows over the scaled feature matrix."""
    seqs = []
    for i in range(len(X_scaled) - seq_len + 1):
        seqs.append(X_scaled[i:i + seq_len])
    return np.array(seqs, dtype=np.float32)


def score_sequences(model, X_scaled, seq_len, batch_size, device):
    """Compute LSTM-AE reconstruction error per window."""
    seqs = make_sequences(X_scaled, seq_len)
    tensor = torch.tensor(seqs)
    errors = np.zeros(len(X_scaled))
    counts = np.zeros(len(X_scaled))

    model.eval()
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            batch = tensor[start:start + batch_size].to(device)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=2).cpu().numpy()
            for b_idx, seq_start in enumerate(range(start, min(start + batch_size, len(seqs)))):
                for t in range(seq_len):
                    w_idx = seq_start + t
                    errors[w_idx] += mse[b_idx, t]
                    counts[w_idx] += 1

    counts = np.maximum(counts, 1)
    return errors / counts


LSTM_SEQ_LEN  = 48   # 48 windows = 48 hours of history (was 24)
LSTM_HIDDEN   = 32
LSTM_EPOCHS   = 30
LSTM_BATCH    = 256
LSTM_LR       = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"LSTM device: {device}")

print("Building LSTM training sequences...", end=" ")
train_seqs = make_sequences(X_train_clean_scaled, LSTM_SEQ_LEN)
train_tensor = torch.tensor(train_seqs)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=LSTM_BATCH, shuffle=True)
print(f"{len(train_seqs):,} sequences of length {LSTM_SEQ_LEN}")

n_features = X_train_clean_scaled.shape[1]
lstm_ae    = LSTMAutoencoder(input_size=n_features, hidden_size=LSTM_HIDDEN).to(device)
optimizer  = torch.optim.Adam(lstm_ae.parameters(), lr=LSTM_LR)
loss_fn    = nn.MSELoss()

print("Training LSTM Autoencoder...")
t0 = time.time()
lstm_ae.train()
for epoch in range(LSTM_EPOCHS):
    epoch_loss = 0.0
    for (batch,) in train_loader:
        batch = batch.to(device)
        recon = lstm_ae(batch)
        loss  = loss_fn(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch)
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:2d}/{LSTM_EPOCHS}  loss={epoch_loss/len(train_seqs):.6f}")
print(f"done ({time.time()-t0:.1f}s)")

lstm_scores_train = score_sequences(lstm_ae, X_train_all_scaled, LSTM_SEQ_LEN, LSTM_BATCH, device)
lstm_scores_val   = score_sequences(lstm_ae, X_val_scaled, LSTM_SEQ_LEN, LSTM_BATCH, device)
lstm_scores_test  = score_sequences(lstm_ae, X_test_scaled, LSTM_SEQ_LEN, LSTM_BATCH, device)

torch.save(lstm_ae.state_dict(), f'{PROJECT_DIR}/models/lstm_autoencoder_v3.pt')


# ============================================================
# SECTION 12c: ANOMALY ENSEMBLE (rank-averaged)
# ============================================================

def rank_normalize(scores):
    return stats.rankdata(scores) / len(scores)


def rank_average(*score_arrays):
    ranks = [rank_normalize(s) for s in score_arrays]
    return np.mean(ranks, axis=0)


ensemble_scores_train = rank_average(iso_scores_train, lof_scores_train, ocsvm_scores_train, lstm_scores_train)
ensemble_scores_val   = rank_average(iso_scores_val, lof_scores_val, ocsvm_scores_val, lstm_scores_val)
ensemble_scores_test  = rank_average(iso_scores_test, lof_scores_test, ocsvm_scores_test, lstm_scores_test)

print(f"Ensemble score ranges:")
print(f"  Train: [{ensemble_scores_train.min():.4f}, {ensemble_scores_train.max():.4f}]")
print(f"  Val:   [{ensemble_scores_val.min():.4f}, {ensemble_scores_val.max():.4f}]")
print(f"  Test:  [{ensemble_scores_test.min():.4f}, {ensemble_scores_test.max():.4f}]")


# ============================================================
# SECTION 13: XGBOOST SUPERVISED CLASSIFIER
# ============================================================
# Combines XRS+SHARP features with anomaly detector scores.

def prepare_supervised_features(X_scaled, iso_scores, lof_scores, ocsvm_scores,
                                 lstm_scores, ensemble_scores):
    """Append anomaly scores as additional features for supervised classifier."""
    return np.column_stack([
        X_scaled,
        iso_scores.reshape(-1, 1),
        lof_scores.reshape(-1, 1),
        ocsvm_scores.reshape(-1, 1),
        lstm_scores.reshape(-1, 1),
        ensemble_scores.reshape(-1, 1),
    ])


X_train_sup = prepare_supervised_features(
    X_train_all_scaled, iso_scores_train, lof_scores_train,
    ocsvm_scores_train, lstm_scores_train, ensemble_scores_train
)
X_val_sup = prepare_supervised_features(
    X_val_scaled, iso_scores_val, lof_scores_val,
    ocsvm_scores_val, lstm_scores_val, ensemble_scores_val
)
X_test_sup = prepare_supervised_features(
    X_test_scaled, iso_scores_test, lof_scores_test,
    ocsvm_scores_test, lstm_scores_test, ensemble_scores_test
)

# Feature names for importance analysis
sup_feature_names = feature_cols + [
    'anomaly_iso', 'anomaly_lof', 'anomaly_ocsvm', 'anomaly_lstm', 'anomaly_ensemble'
]

n_pos = y_train_all.sum()
n_neg = len(y_train_all) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)
print(f"Class balance: {n_neg:,} neg / {n_pos:,} pos  |  scale_pos_weight={scale_pos_weight:.1f}")

print("Training XGBoost classifier...")
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='aucpr',
    early_stopping_rounds=30,
    random_state=42,
    use_label_encoder=False,
    tree_method='hist',
)
xgb_model.fit(
    X_train_sup, y_train_all,
    sample_weight=w_train_all,
    eval_set=[(X_val_sup, y_val)],
    verbose=10
)
print(f"XGBoost done ({time.time()-t0:.1f}s)  |  best iteration: {xgb_model.best_iteration}")

# Predict probabilities
xgb_probs_val  = xgb_model.predict_proba(X_val_sup)[:, 1]
xgb_probs_test = xgb_model.predict_proba(X_test_sup)[:, 1]

# Feature importance
importance = xgb_model.feature_importances_
imp_df = pd.DataFrame({'feature': sup_feature_names, 'importance': importance})
imp_df = imp_df.sort_values('importance', ascending=False)
print("\nTop 20 features by importance:")
print(imp_df.head(20).to_string(index=False))


# ============================================================
# SECTION 14: EVALUATION
# ============================================================

def apply_temporal_filter(predictions, min_hours=2):
    """
    Post-prediction temporal consistency filter.
    Require min_hours consecutive positive predictions before raising alert.
    """
    filtered = np.zeros(len(predictions), dtype=int)
    consec = 0
    for i, p in enumerate(predictions):
        if p == 1:
            consec += 1
            if consec >= min_hours:
                filtered[i] = 1
        else:
            consec = 0
    return filtered


def evaluate_comprehensive(scores, feature_df_subset, flare_catalog,
                           threshold=None, pre_flare_hours=24,
                           name="Detector", temporal_filter_hours=2):
    """
    Comprehensive evaluation with event-level as primary metric.

    If threshold is None, uses Youden's J on the provided data.
    Reports: event_recall, event_precision, window metrics, HSS, FAR,
             lead time, ROC AUC, PR AUC.
    """
    test_starts   = feature_df_subset['window_start'].values
    test_ends     = feature_df_subset['window_end'].values
    window_labels = np.zeros(len(scores))

    for _, flare in flare_catalog.iterrows():
        flare_start = pd.Timestamp(flare['start_time'])
        pre_start   = flare_start - pd.Timedelta(hours=pre_flare_hours)
        in_zone = (
            (test_starts >= np.datetime64(pre_start)) &
            (test_ends   <= np.datetime64(flare_start))
        )
        window_labels[in_zone] = 1

    # ROC AUC and PR AUC
    n_unique = len(np.unique(window_labels))
    roc_auc = roc_auc_score(window_labels, scores) if n_unique > 1 else 0.0
    pr_auc  = average_precision_score(window_labels, scores) if n_unique > 1 else 0.0

    # Threshold selection: Youden's J on provided data
    if threshold is None:
        if n_unique > 1:
            fpr, tpr, thresholds = roc_curve(window_labels, scores)
            best_idx  = np.argmax(tpr - fpr)
            threshold = thresholds[best_idx]
        else:
            threshold = np.percentile(scores, 95)

    # Binary predictions (NO rolling median smoothing — raw threshold)
    raw_preds = (scores >= threshold).astype(int)

    # Temporal consistency filter (post-prediction only)
    predictions = apply_temporal_filter(raw_preds, min_hours=temporal_filter_hours)

    # --- Event-level metrics ---
    flares_detected = 0
    lead_times = []
    for _, flare in flare_catalog.iterrows():
        flare_start = pd.Timestamp(flare['start_time'])
        pre_start   = flare_start - pd.Timedelta(hours=pre_flare_hours)
        in_zone = (
            (test_starts >= np.datetime64(pre_start)) &
            (test_ends   <= np.datetime64(flare_start))
        )
        zone_preds  = predictions[in_zone]
        zone_starts = test_starts[in_zone]
        if zone_preds.sum() > 0:
            flares_detected += 1
            first_alert = pd.Timestamp(zone_starts[np.where(zone_preds == 1)[0][0]])
            lead_times.append((flare_start - first_alert).total_seconds() / 3600)

    n_flares     = len(flare_catalog)
    event_recall = flares_detected / n_flares if n_flares > 0 else 0

    # --- Window-level metrics ---
    tp = ((predictions == 1) & (window_labels == 1)).sum()
    fp = ((predictions == 1) & (window_labels == 0)).sum()
    fn = ((predictions == 0) & (window_labels == 1)).sum()
    tn = ((predictions == 0) & (window_labels == 0)).sum()

    precision     = tp / (tp + fp) if (tp + fp) > 0 else 0
    window_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * window_recall / (precision + window_recall)
          if (precision + window_recall) > 0 else 0)

    # HSS (Heidke Skill Score)
    N = tp + fp + fn + tn
    expected = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / N if N > 0 else 0
    hss = (tp + tn - expected) / (N - expected) if (N - expected) > 0 else 0

    # TSS (True Skill Statistic)
    tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    tss = tpr_val - fpr_val

    # False alarm rate (FP per day)
    total_hours = len(scores)  # 1 window per hour
    far_per_day = (fp / total_hours * 24) if total_hours > 0 else 0

    avg_lead  = np.mean(lead_times) if lead_times else 0
    med_lead  = np.median(lead_times) if lead_times else 0

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  EVENT RECALL (primary): {flares_detected} / {n_flares}  ({event_recall:.4f})")
    print(f"  Window Recall:   {window_recall:.4f}")
    print(f"  Precision:       {precision:.4f}")
    print(f"  F1 (window):     {f1:.4f}")
    print(f"  ROC AUC:         {roc_auc:.4f}")
    print(f"  PR AUC:          {pr_auc:.4f}")
    print(f"  HSS:             {hss:.4f}")
    print(f"  TSS:             {tss:.4f}")
    print(f"  False alarms/day:{far_per_day:.2f}")
    print(f"  Threshold:       {threshold:.4f}")
    print(f"  Avg Lead Time:   {avg_lead:.1f} hours")
    print(f"  Median Lead Time:{med_lead:.1f} hours")

    return dict(
        detector=name, event_recall=event_recall, window_recall=window_recall,
        precision=precision, f1=f1, roc_auc=roc_auc, pr_auc=pr_auc,
        hss=hss, tss=tss, far_per_day=far_per_day,
        avg_lead_time_hours=avg_lead, median_lead_time_hours=med_lead,
        flares_detected=flares_detected, total_flares=n_flares,
        threshold=threshold
    )


# --- Optimize XGBoost threshold on validation set ---
print("\n" + "="*60)
print("  THRESHOLD OPTIMIZATION ON VALIDATION SET (2020–2021)")
print("="*60)

feature_df_val = feature_df.loc[val_mask].copy()

# Find threshold that maximizes event F1 with event_recall >= 0.6
best_event_f1 = 0
best_threshold = 0.5
for pct in range(5, 96):
    thr = np.percentile(xgb_probs_val, pct)
    raw_preds = (xgb_probs_val >= thr).astype(int)
    preds = apply_temporal_filter(raw_preds, min_hours=2)

    detected = 0
    for _, flare in val_flares.iterrows():
        fs = pd.Timestamp(flare['start_time'])
        ps = fs - pd.Timedelta(hours=24)
        starts = feature_df_val['window_start'].values
        ends   = feature_df_val['window_end'].values
        in_zone = (starts >= np.datetime64(ps)) & (ends <= np.datetime64(fs))
        if preds[in_zone].sum() > 0:
            detected += 1

    if len(val_flares) > 0:
        e_recall = detected / len(val_flares)
    else:
        continue

    # Total positive predictions as proxy for precision calc
    total_pos = preds.sum()
    if total_pos > 0 and e_recall >= 0.5:  # relaxed from 0.6 for flexibility
        e_f1_proxy = 2 * e_recall * (detected / total_pos) / (e_recall + detected / total_pos) \
                     if (e_recall + detected / total_pos) > 0 else 0
        if e_f1_proxy > best_event_f1:
            best_event_f1 = e_f1_proxy
            best_threshold = thr

# Fallback to Youden's J if no threshold meets constraint
if best_threshold == 0.5 and len(np.unique(y_val)) > 1:
    fpr_v, tpr_v, thr_v = roc_curve(y_val, xgb_probs_val)
    best_threshold = thr_v[np.argmax(tpr_v - fpr_v)]

print(f"Optimized threshold: {best_threshold:.4f}")


# --- Final evaluation on TEST set (2022–2024) ---
print("\n" + "="*60)
print("  FINAL EVALUATION ON TEST SET (2022–2024)")
print("="*60)

feature_df_test = feature_df.loc[test_mask].copy()
test_flares = mx_flares[pd.to_datetime(mx_flares['start_time']) >= VAL_END]

results = []

# Baselines
np.random.seed(42)
random_scores = np.random.rand(len(ensemble_scores_test))
results.append(evaluate_comprehensive(random_scores, feature_df_test, test_flares,
                                       name="Random Baseline"))

# Individual anomaly detectors
results.append(evaluate_comprehensive(iso_scores_test, feature_df_test, test_flares,
                                       name="Isolation Forest"))
results.append(evaluate_comprehensive(lof_scores_test, feature_df_test, test_flares,
                                       name="Local Outlier Factor"))
results.append(evaluate_comprehensive(ocsvm_scores_test, feature_df_test, test_flares,
                                       name="One-Class SVM"))
results.append(evaluate_comprehensive(lstm_scores_test, feature_df_test, test_flares,
                                       name="LSTM Autoencoder"))

# Anomaly ensemble
results.append(evaluate_comprehensive(ensemble_scores_test, feature_df_test, test_flares,
                                       name="Anomaly Ensemble"))

# XGBoost (primary model) — threshold from validation
results.append(evaluate_comprehensive(xgb_probs_test, feature_df_test, test_flares,
                                       threshold=best_threshold,
                                       name="XGBoost + Anomaly Ensemble (PRIMARY)"))

results_df = pd.DataFrame(results)
print(f"\n{'='*70}")
print(f"  RESULTS SUMMARY — TEST SET (2022–2024)")
print(f"{'='*70}")
print(results_df[['detector', 'event_recall', 'window_recall', 'precision',
                   'f1', 'roc_auc', 'pr_auc', 'hss', 'tss',
                   'avg_lead_time_hours']].to_string(index=False))
results_df.to_csv(f'{RESULTS_DIR}/v3_results.csv', index=False)


# ============================================================
# SECTION 14b: M-CLASS vs X-CLASS BREAKDOWN
# ============================================================
m_flares_test = test_flares[test_flares['class_letter'] == 'M']
x_flares_test = test_flares[test_flares['class_letter'] == 'X']

print(f"\nTest flares: {len(m_flares_test)} M-class, {len(x_flares_test)} X-class")

mx_results = []
for scores, det_name in [
    (ensemble_scores_test, "Anomaly Ensemble"),
    (xgb_probs_test, "XGBoost (PRIMARY)"),
]:
    for cls_label, subset in [("M", m_flares_test), ("X", x_flares_test)]:
        if len(subset) == 0:
            continue
        thr = best_threshold if "XGBoost" in det_name else None
        r = evaluate_comprehensive(scores, feature_df_test, subset,
                                    threshold=thr,
                                    name=f"{det_name} — {cls_label}-class")
        mx_results.append(r)

mx_df = pd.DataFrame(mx_results)
print(f"\n{'='*70}")
print(f"  M vs X CLASS BREAKDOWN")
print(f"{'='*70}")
print(mx_df[['detector', 'event_recall', 'window_recall', 'precision',
             'f1', 'roc_auc', 'avg_lead_time_hours']].to_string(index=False))
mx_df.to_csv(f'{RESULTS_DIR}/v3_mx_class_results.csv', index=False)


# ============================================================
# SECTION 15: VISUALIZATIONS
# ============================================================

# --- 15a: Anomaly score + XGBoost probability timeline ---
fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

test_times = pd.to_datetime(feature_df_test['window_start'].values)

axes[0].plot(test_times, xgb_probs_test, linewidth=0.3, color='steelblue', alpha=0.7)
axes[0].axhline(best_threshold, color='red', linestyle='--', alpha=0.7,
                linewidth=1, label=f"Threshold={best_threshold:.3f}")
for ft in pd.to_datetime(test_flares['start_time']):
    axes[0].axvline(ft, color='orange', alpha=0.15, linewidth=0.5)
axes[0].set_ylabel('XGBoost P(flare)')
axes[0].set_title('XGBoost Predictions vs Actual M/X Flares (orange lines) — Test 2022–2024')
axes[0].legend(loc='upper right')

axes[1].plot(test_times, ensemble_scores_test, linewidth=0.3, color='darkgreen', alpha=0.7)
for ft in pd.to_datetime(test_flares['start_time']):
    axes[1].axvline(ft, color='orange', alpha=0.15, linewidth=0.5)
axes[1].set_ylabel('Anomaly Ensemble Score')
axes[1].set_title('Anomaly Ensemble Scores')

test_flux = df.loc[df.index >= str(VAL_END), 'xrsb']
axes[2].semilogy(test_flux.index, test_flux, linewidth=0.15, color='navy', alpha=0.6)
axes[2].axhline(1e-5, color='orange', linestyle='--', alpha=0.4, label='M-class')
axes[2].axhline(1e-4, color='red',    linestyle='--', alpha=0.4, label='X-class')
axes[2].set_ylabel('XRS-B Flux (W/m²)')
axes[2].set_xlabel('Date')
axes[2].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/v3_predictions_timeline.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 15b: Feature importance ---
fig, ax = plt.subplots(figsize=(10, 8))
top_n = min(25, len(imp_df))
top_imp = imp_df.head(top_n).iloc[::-1]
ax.barh(top_imp['feature'], top_imp['importance'], color='steelblue')
ax.set_xlabel('Feature Importance (gain)')
ax.set_title('XGBoost Top Feature Importances')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/v3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 15c: Score distributions ---
feature_df_test_starts = feature_df_test['window_start'].values
window_labels_test = np.zeros(len(xgb_probs_test))
for _, flare in test_flares.iterrows():
    flare_start = pd.Timestamp(flare['start_time'])
    pre_start   = flare_start - pd.Timedelta(hours=24)
    in_zone = (
        (feature_df_test_starts >= np.datetime64(pre_start)) &
        (feature_df_test_starts <= np.datetime64(flare_start))
    )
    window_labels_test[in_zone] = 1

quiet_scores    = xgb_probs_test[window_labels_test == 0]
preflare_scores = xgb_probs_test[window_labels_test == 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(quiet_scores,    bins=50, alpha=0.6, density=True, color='steelblue',
             label=f'Quiet ({len(quiet_scores):,})')
axes[0].hist(preflare_scores, bins=50, alpha=0.6, density=True, color='coral',
             label=f'Pre-flare ({len(preflare_scores):,})')
axes[0].axvline(best_threshold, color='red', linestyle='--', alpha=0.7,
                label=f"Threshold={best_threshold:.3f}")
axes[0].set_xlabel('XGBoost P(flare)')
axes[0].set_ylabel('Density')
axes[0].set_title('Score Distributions: Quiet vs Pre-Flare')
axes[0].legend()

bp = axes[1].boxplot([quiet_scores, preflare_scores],
                      tick_labels=['Quiet', 'Pre-Flare'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_ylabel('XGBoost P(flare)')
axes[1].set_title('Score Comparison')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/v3_score_distributions.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 15d: Calibration curve ---
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(8, 8))
prob_true, prob_pred = calibration_curve(window_labels_test, xgb_probs_test,
                                          n_bins=15, strategy='quantile')
ax.plot(prob_pred, prob_true, 's-', color='steelblue', label='XGBoost')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Reliability Diagram')
ax.legend()
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/v3_calibration.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 15e: Spot-check first true positive ---
predictions_test = apply_temporal_filter(
    (xgb_probs_test >= best_threshold).astype(int), min_hours=2
)
test_starts_arr = feature_df_test['window_start'].values

found_tp = False
for _, flare in test_flares.iterrows():
    flare_start = pd.Timestamp(flare['start_time'])
    pre_start   = flare_start - pd.Timedelta(hours=24)
    in_zone = (
        (test_starts_arr >= np.datetime64(pre_start)) &
        (test_starts_arr <= np.datetime64(flare_start))
    )
    if predictions_test[in_zone].sum() > 0:
        plot_start = flare_start - pd.Timedelta(hours=30)
        plot_end   = flare_start + pd.Timedelta(hours=6)
        mask = (df.index >= plot_start) & (df.index <= plot_end)

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        axes[0].semilogy(df.loc[mask].index, df.loc[mask, 'xrsb'], linewidth=0.8, color='navy')
        axes[0].axvline(flare_start, color='red',    linestyle='--', linewidth=2, label='Flare start')
        axes[0].axvline(pre_start,   color='orange', linestyle=':',  linewidth=1.5, label='24h window')
        axes[0].set_ylabel('XRS-B Flux (W/m²)')
        axes[0].set_title(f"TRUE POSITIVE: {flare['flare_class']} at {flare_start}")
        axes[0].legend()

        score_mask  = ((test_starts_arr >= np.datetime64(plot_start)) &
                       (test_starts_arr <= np.datetime64(plot_end)))
        score_times = pd.to_datetime(test_starts_arr[score_mask])
        score_vals  = xgb_probs_test[score_mask]

        axes[1].plot(score_times, score_vals, linewidth=1.5, color='steelblue',
                     marker='o', markersize=2)
        axes[1].axhline(best_threshold, color='red', linestyle='--', alpha=0.7,
                        label=f"Threshold={best_threshold:.3f}")
        axes[1].axvline(flare_start, color='red',    linestyle='--', linewidth=2)
        axes[1].axvline(pre_start,   color='orange', linestyle=':',  linewidth=1.5)
        axes[1].set_ylabel('XGBoost P(flare)')
        axes[1].set_xlabel('Time')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/v3_true_positive_example.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Flare: {flare['flare_class']} at {flare_start}")
        found_tp = True
        break

if not found_tp:
    print("No true positives found — check evaluation.")


# ============================================================
# FINAL SUMMARY
# ============================================================
xgb_row = results_df[results_df['detector'].str.contains('PRIMARY')].iloc[0]
ens_row = results_df[results_df['detector'] == 'Anomaly Ensemble'].iloc[0]

print(f"\n{'='*60}")
print(f"  PIPELINE v3 — COMPLETE")
print(f"{'='*60}")
print(f"  Data:      GOES-15/16 XRS + SDO/HMI SHARP, 2012–2024")
print(f"  Rows:      {len(df):,} (1-minute cadence)")
print(f"  Flares:    {len(mx_flares)} M/X class")
print(f"  Windows:   {len(feature_df):,} ({len(feature_cols)} features each)")
print(f"  Train:     {train_mask.sum():,} windows (2012–2019)")
print(f"  Val:       {val_mask.sum():,} windows (2020–2021)")
print(f"  Test:      {test_mask.sum():,} windows (2022–2024)")
print(f"")
print(f"  Anomaly Detectors: IF, LOF, OCSVM, LSTM-AE (48h lookback)")
print(f"  Primary Model:     XGBoost on {len(sup_feature_names)} features")
print(f"                     (XRS relative + SHARP + anomaly scores)")
print(f"  Threshold:         {best_threshold:.4f} (optimized on validation)")
print(f"  Temporal filter:   2-hour consistency (post-prediction)")
print(f"")
print(f"  RESULTS — Anomaly Ensemble (baseline):")
print(f"    Event Recall:   {ens_row['event_recall']:.3f}")
print(f"    Window Recall:  {ens_row['window_recall']:.3f}")
print(f"    Precision:      {ens_row['precision']:.3f}")
print(f"    F1 (window):    {ens_row['f1']:.3f}")
print(f"    ROC AUC:        {ens_row['roc_auc']:.3f}")
print(f"")
print(f"  RESULTS — XGBoost + Anomaly Ensemble (PRIMARY):")
print(f"    Event Recall:   {xgb_row['event_recall']:.3f}")
print(f"    Window Recall:  {xgb_row['window_recall']:.3f}")
print(f"    Precision:      {xgb_row['precision']:.3f}")
print(f"    F1 (window):    {xgb_row['f1']:.3f}")
print(f"    ROC AUC:        {xgb_row['roc_auc']:.3f}")
print(f"    PR AUC:         {xgb_row['pr_auc']:.3f}")
print(f"    HSS:            {xgb_row['hss']:.3f}")
print(f"    TSS:            {xgb_row['tss']:.3f}")
print(f"    Lead Time:      {xgb_row['avg_lead_time_hours']:.1f} hours (avg)")
print(f"")
print(f"  FILES SAVED:")
print(f"    data/processed/goes_xrs_2012_2024.csv")
print(f"    data/features/v3_features.csv")
print(f"    data/flare_catalog/mx_flares_2012_2024.csv")
print(f"    data/sharp/sharp_aggregated_2012_2024.csv")
print(f"    models/lstm_autoencoder_v3.pt")
print(f"    results/v3_results.csv")
print(f"    results/v3_mx_class_results.csv")
print(f"    results/v3_predictions_timeline.png")
print(f"    results/v3_feature_importance.png")
print(f"    results/v3_score_distributions.png")
print(f"    results/v3_calibration.png")
print(f"    results/v3_true_positive_example.png")
print(f"{'='*60}")
