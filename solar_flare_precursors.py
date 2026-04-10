# ============================================================
# Solar Flare Precursor Detection — Baseline Pipeline
# GOES-16 XRS Anomaly Detection (2017–2024)
# ============================================================

# ============================================================
# SETUP
# ============================================================
# !pip install netCDF4 xarray "sunpy[net]" mpl_animators torch

from google.colab import drive
drive.mount('/content/drive')

import os

PROJECT_DIR = '/content/drive/MyDrive/solar-flare-precursors'

folders = [
    f'{PROJECT_DIR}/data/raw',
    f'{PROJECT_DIR}/data/processed',
    f'{PROJECT_DIR}/data/features',
    f'{PROJECT_DIR}/data/flare_catalog',
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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# SECTION 1: DOWNLOAD GOES XRS DATA (2017–2024)
# ============================================================
# Downloads year-by-year so interruptions don't lose progress.

years = range(2017, 2025)

for year in years:
    year_files = glob.glob(f'{RAW_DIR}/*{year}*.nc') + \
                 glob.glob(f'{RAW_DIR}/*{year}*.fits')
    if len(year_files) > 300:
        print(f"  {year}: Already have {len(year_files)} files — skipping.")
        continue

    print(f"\nDownloading {year}...")
    try:
        result = Fido.search(
            a.Time(f'{year}-01-01', f'{year}-12-31'),
            a.Instrument.xrs,
            a.goes.SatelliteNumber(16),
            a.Resolution("avg1m")
        )
        files = Fido.fetch(result, path=f'{RAW_DIR}/{{file}}')
        print(f"  Downloaded {len(files)} files for {year}")
    except Exception as e:
        print(f"  ERROR downloading {year}: {e}")
        continue

print(f"Total raw files: {len(glob.glob(f'{RAW_DIR}/*.nc') + glob.glob(f'{RAW_DIR}/*.fits'))}")


# ============================================================
# SECTION 2: LOAD ALL DATA INTO ONE DATAFRAME
# ============================================================
SAVE_PATH = f'{PROCESSED_DIR}/goes_xrs_2017_2024.csv'

if os.path.exists(SAVE_PATH):
    print(f"Loading processed file from Drive...")
    df = pd.read_csv(SAVE_PATH, parse_dates=['timestamp'], index_col='timestamp')
else:
    files = sorted(glob.glob(f'{RAW_DIR}/*.nc') + glob.glob(f'{RAW_DIR}/*.fits'))
    print(f"Found {len(files)} raw files. Loading year by year...")

    all_dfs = []
    for year in range(2017, 2025):
        year_files = [f for f in files if str(year) in os.path.basename(f)]
        if not year_files:
            continue
        try:
            ts = TimeSeries(year_files, concatenate=True)
            year_df = ts.to_dataframe()
            all_dfs.append(year_df)
            print(f"  {year}: {len(year_df):,} rows")
        except Exception as e:
            print(f"  {year}: ERROR — {e}")

    df = pd.concat(all_dfs).sort_index()
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        df = df[~df.index.duplicated(keep='first')]
    df.index.name = 'timestamp'
    df.to_csv(SAVE_PATH)
    print(f"Saved to: {SAVE_PATH}")

print(f"Shape: {df.shape}  |  {df.index.min()} to {df.index.max()}")


# ============================================================
# SECTION 3: DATA CLEANING
# ============================================================
# Step 1: Quality flag (0 = good data)
bad_quality = (df['xrsb_quality'] != 0) | (df['xrsa_quality'] != 0)
df.loc[bad_quality, ['xrsb', 'xrsa']] = np.nan

# Step 2: Negative values (instrument artifacts)
df.loc[df['xrsb'] < 0, 'xrsb'] = np.nan
df.loc[df['xrsa'] < 0, 'xrsa'] = np.nan

# Step 3: Zeros (data gaps disguised as data)
df.loc[df['xrsb'] == 0, 'xrsb'] = np.nan
df.loc[df['xrsa'] == 0, 'xrsa'] = np.nan

# Step 4: Interpolate short gaps only (< 10 minutes)
df['xrsb'] = df['xrsb'].interpolate(method='linear', limit=10)
df['xrsa'] = df['xrsa'].interpolate(method='linear', limit=10)

# Step 5: Channel ratio (temperature proxy)
df['channel_ratio'] = df['xrsa'] / df['xrsb']
df['channel_ratio'] = df['channel_ratio'].replace([np.inf, -np.inf], np.nan)
df['channel_ratio'] = df['channel_ratio'].interpolate(method='linear', limit=10)

# Step 6: Mark remaining gaps
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
axes[0].set_title('GOES-16 X-ray Flux — Full Dataset (2017–2024)')
axes[0].legend(loc='upper left', fontsize=8)

axes[1].semilogy(df.index, df['xrsa'], linewidth=0.15, color='darkred', alpha=0.6)
axes[1].set_ylabel('XRS-A (0.5–4 Å)\nW/m²')

axes[2].plot(df.index, df['channel_ratio'], linewidth=0.15, color='darkgreen', alpha=0.6)
axes[2].set_ylabel('Channel Ratio\n(A / B)')
axes[2].set_xlabel('Date')
axes[2].set_ylim(0, 2)

split_date = pd.Timestamp('2020-01-01')
for ax in axes:
    ax.axvline(split_date, color='red', linestyle='-', linewidth=1.5, alpha=0.8)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/full_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# SECTION 5: DOWNLOAD FLARE CATALOG (M/X CLASS, 2017–2024)
# ============================================================
CATALOG_PATH = f'{CATALOG_DIR}/mx_flares_2017_2024.csv'

if os.path.exists(CATALOG_PATH):
    print("Loading flare catalog from Drive...")
    mx_flares = pd.read_csv(CATALOG_PATH, parse_dates=['start_time', 'peak_time', 'end_time'])
else:
    client = hek.HEKClient()
    all_flares = []

    for year in range(2017, 2025):
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
# SECTION 6: VISUALIZE FLARE DISTRIBUTION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

mx_flares['year'] = mx_flares['start_time'].dt.year
yearly = mx_flares.groupby(['year', 'class_letter']).size().unstack(fill_value=0)
yearly.plot(kind='bar', ax=axes[0], color={'M': 'steelblue', 'X': 'coral'},
            edgecolor='black', linewidth=0.5)
axes[0].set_title('M/X Flares Per Year')
axes[0].set_ylabel('Count')
axes[0].axvline(2.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].tick_params(axis='x', rotation=45)

m_flares = mx_flares[mx_flares['class_letter'] == 'M']
x_flares_only = mx_flares[mx_flares['class_letter'] == 'X']
axes[1].scatter(m_flares['start_time'], [1]*len(m_flares), s=5, alpha=0.4,
                color='steelblue', label=f'M-class ({len(m_flares)})')
axes[1].scatter(x_flares_only['start_time'], [2]*len(x_flares_only), s=20, alpha=0.7,
                color='coral', label=f'X-class ({len(x_flares_only)})')
axes[1].axvline(pd.Timestamp('2020-01-01'), color='red', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].set_yticks([1, 2])
axes[1].set_yticklabels(['M', 'X'])
axes[1].set_title('Flare Timeline')
axes[1].legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/flare_distribution.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# SECTION 7: FEATURE ENGINEERING
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


def compute_baseline_features(xrsb, xrsa, ratio):
    """
    Computes 20 features for a single window:
      - 9 time-domain  (mean, std, skew, kurtosis, min, max, range, slope, max_deriv)
      - 6 frequency-domain  (dominant_freq/power, spectral_entropy, total_power, centroid, rolloff)
      - 2 complexity  (permutation_entropy, sample_entropy)
      - 3 multi-channel  (ratio_mean, ratio_std, ratio_slope)
    Returns None if >20% of the window is NaN.
    """
    if np.isnan(xrsb).sum() > len(xrsb) * 0.2:
        return None

    xrsb  = pd.Series(xrsb).ffill().bfill().values
    xrsa  = pd.Series(xrsa).ffill().bfill().values
    ratio = pd.Series(ratio).ffill().bfill().values

    log_b = np.log10(np.clip(xrsb, 1e-10, None))
    log_a = np.log10(np.clip(xrsa, 1e-10, None))

    features = {}

    # --- Time-domain ---
    features['xrsb_mean']         = np.mean(log_b)
    features['xrsb_std']          = np.std(log_b)
    features['xrsb_skew']         = stats.skew(log_b)
    features['xrsb_kurtosis']     = stats.kurtosis(log_b)
    features['xrsb_min']          = np.min(log_b)
    features['xrsb_max']          = np.max(log_b)
    slope, *_ = stats.linregress(np.arange(len(log_b)), log_b)
    features['xrsb_slope']        = slope
    diffs = np.diff(log_b)
    features['xrsb_max_abs_deriv'] = np.max(np.abs(diffs)) if len(diffs) > 0 else 0

    # --- Frequency-domain ---
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
    features['dominant_power']   = np.max(power) if len(power) > 0 else 0
    pn_safe = power_norm[power_norm > 0]
    features['spectral_entropy'] = -np.sum(pn_safe * np.log2(pn_safe)) if len(pn_safe) > 0 else 0
    features['total_power']      = power_sum
    features['spectral_centroid'] = np.sum(xf * power) / power_sum if power_sum > 0 else 0
    cum_power = np.cumsum(power)
    rolloff_idx = np.searchsorted(cum_power, 0.85 * cum_power[-1]) if len(cum_power) > 0 else 0
    features['spectral_rolloff'] = xf[min(rolloff_idx, len(xf)-1)] if len(xf) > 0 else 0

    # --- Complexity ---
    features['permutation_entropy'] = _permutation_entropy(log_b, order=3, delay=1)
    step = max(1, len(log_b) // 60)
    features['sample_entropy'] = _sample_entropy(log_b[::step], m=2, r=0.2*np.std(log_b[::step]))

    # --- Multi-channel ---
    features['ratio_mean']  = np.mean(ratio)
    features['ratio_std']   = np.std(ratio)
    ratio_slope, *_ = stats.linregress(np.arange(len(ratio)), ratio)
    features['ratio_slope'] = ratio_slope

    return features


# ============================================================
# SECTION 8: EXTRACT FEATURES ACROSS FULL DATASET
# ============================================================
FEATURES_PATH = f'{FEATURES_DIR}/baseline_features.csv'

if os.path.exists(FEATURES_PATH):
    print("Loading feature matrix from Drive...")
    feature_df = pd.read_csv(FEATURES_PATH, parse_dates=['window_start', 'window_end'])
else:
    window_size = 360   # 6 hours
    stride      = 60    # 1 hour

    n_windows = (len(df) - window_size) // stride + 1
    print(f"Processing {n_windows:,} windows (this takes 15-30 min)...")

    all_features = []
    skipped = 0
    start_time = time.time()
    ten_pct = max(1, n_windows // 10)

    for i in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[i:i + window_size]
        feats = compute_baseline_features(
            window['xrsb'].values, window['xrsa'].values, window['channel_ratio'].values
        )
        if feats is not None:
            feats['window_start'] = window.index[0]
            feats['window_end']   = window.index[-1]
            all_features.append(feats)
        else:
            skipped += 1

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
# SECTION 8b: CROSS-WINDOW DELTA FEATURES
# ============================================================
# These capture how the flux background is changing *between* windows —
# the most useful pre-flare signal in XRS data. Computed after the window
# loop so they can reference prior rows without re-engineering the feature fn.

feature_df = feature_df.sort_values('window_start').reset_index(drop=True)

for lag_h in [1, 6, 24]:
    lag_rows = lag_h  # stride = 1h, so 1 lag_row = 1 hour
    feature_df[f'delta_mean_{lag_h}h'] = (
        feature_df['xrsb_mean'] - feature_df['xrsb_mean'].shift(lag_rows)
    )
    feature_df[f'delta_std_{lag_h}h'] = (
        feature_df['xrsb_std'] - feature_df['xrsb_std'].shift(lag_rows)
    )

# 24-hour background trend slope (slope of hourly xrsb_mean over prior 24 windows)
feature_df['background_trend_24h'] = (
    feature_df['xrsb_mean']
    .rolling(24, min_periods=6)
    .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
)

delta_cols = [c for c in feature_df.columns if c.startswith('delta_') or c == 'background_trend_24h']
feature_cols = feature_cols + delta_cols
print(f"Added {len(delta_cols)} cross-window delta features → {len(feature_cols)} total")


# ============================================================
# SECTION 9: FEATURE INSPECTION & CLEANUP
# ============================================================
feature_df['window_start'] = pd.to_datetime(feature_df['window_start'])
feature_df['window_end']   = pd.to_datetime(feature_df['window_end'])

n_inf = np.isinf(feature_df[feature_cols].values).sum()
if n_inf > 0:
    print(f"Replacing {n_inf} Inf values with NaN (imputation deferred to Section 10)")
    feature_df[feature_cols] = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan)
# NaN imputation is done in Section 10 with SimpleImputer fit on training data only.

print(feature_df[feature_cols].describe().round(4).to_string())


# ============================================================
# SECTION 10: TRAIN/TEST SPLIT WITH FLARE EXCLUSION
# ============================================================
# Training: 2017–2019, excluding any window within 24h before an M/X flare.
# Test: 2020–2024 (everything, including pre-flare windows).

split_date  = pd.Timestamp('2020-01-01')
train_mask  = feature_df['window_end'] < split_date
test_mask   = feature_df['window_start'] >= split_date

train_flares     = mx_flares[mx_flares['start_time'] < split_date]
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

train_clean = train_mask & ~exclusion_mask
print(f"Training windows: {train_clean.sum():,}  |  Test windows: {test_mask.sum():,}")

X_train = feature_df.loc[train_clean, feature_cols].values
X_test  = feature_df.loc[test_mask,   feature_cols].values

# Impute with training-set medians only (no leakage from test)
imputer        = SimpleImputer(strategy='median')
X_train        = imputer.fit_transform(X_train)
X_test         = imputer.transform(X_test)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# Safety net for any zero-variance columns that survive scaling
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
X_test_scaled  = np.nan_to_num(X_test_scaled,  nan=0, posinf=0, neginf=0)


# ============================================================
# SECTION 11: TRAIN ANOMALY DETECTORS
# ============================================================

# Isolation Forest
print("Training Isolation Forest...", end=" ")
t0 = time.time()
iso_forest = IsolationForest(n_estimators=200, contamination='auto', random_state=42, n_jobs=-1)
iso_forest.fit(X_train_scaled)
iso_scores_test = -iso_forest.score_samples(X_test_scaled)
print(f"done ({time.time()-t0:.1f}s)  range: [{iso_scores_test.min():.4f}, {iso_scores_test.max():.4f}]")

# Local Outlier Factor
print("Training Local Outlier Factor...", end=" ")
t0 = time.time()
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto', n_jobs=-1)
lof.fit(X_train_scaled)
lof_scores_test = -lof.score_samples(X_test_scaled)
print(f"done ({time.time()-t0:.1f}s)  range: [{lof_scores_test.min():.4f}, {lof_scores_test.max():.4f}]")

# One-Class SVM
print("Training One-Class SVM...", end=" ")
t0 = time.time()
ocsvm = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
ocsvm.fit(X_train_scaled)
ocsvm_scores_test = -ocsvm.score_samples(X_test_scaled)
print(f"done ({time.time()-t0:.1f}s)  range: [{ocsvm_scores_test.min():.4f}, {ocsvm_scores_test.max():.4f}]")

# ============================================================
# SECTION 11b: LSTM AUTOENCODER
# ============================================================
# Learns what normal *sequences* of solar activity look like.
# Anomaly score = reconstruction error on unseen test sequences.

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        # Repeat latent state across sequence length for decoding
        repeated = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _   = self.decoder(repeated)
        return out


def make_sequences(X_scaled, seq_len=24):
    """Slide a window of seq_len rows over the scaled feature matrix."""
    seqs = []
    for i in range(len(X_scaled) - seq_len + 1):
        seqs.append(X_scaled[i:i + seq_len])
    return np.array(seqs, dtype=np.float32)


LSTM_SEQ_LEN  = 24   # 24 windows = 24 hours of history per sequence
LSTM_HIDDEN   = 32
LSTM_EPOCHS   = 30
LSTM_BATCH    = 256
LSTM_LR       = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"LSTM device: {device}")

print("Building LSTM training sequences...", end=" ")
train_seqs = make_sequences(X_train_scaled, LSTM_SEQ_LEN)
train_tensor = torch.tensor(train_seqs)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=LSTM_BATCH, shuffle=True)
print(f"{len(train_seqs):,} sequences of length {LSTM_SEQ_LEN}")

n_features  = X_train_scaled.shape[1]
lstm_ae     = LSTMAutoencoder(input_size=n_features, hidden_size=LSTM_HIDDEN).to(device)
optimizer   = torch.optim.Adam(lstm_ae.parameters(), lr=LSTM_LR)
loss_fn     = nn.MSELoss()

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

# Score test sequences: reconstruction MSE per sequence → per window
# Each window i gets the score from the sequence ending at window i.
print("Scoring test sequences...", end=" ")
lstm_ae.eval()
test_seqs   = make_sequences(X_test_scaled, LSTM_SEQ_LEN)
test_tensor = torch.tensor(test_seqs)
recon_errors = np.zeros(len(X_test_scaled))
recon_counts = np.zeros(len(X_test_scaled))

with torch.no_grad():
    for start in range(0, len(test_seqs), LSTM_BATCH):
        batch = test_tensor[start:start + LSTM_BATCH].to(device)
        recon = lstm_ae(batch)
        mse   = ((recon - batch) ** 2).mean(dim=2).cpu().numpy()  # (batch, seq_len)
        for b_idx, seq_start in enumerate(range(start, min(start + LSTM_BATCH, len(test_seqs)))):
            for t in range(LSTM_SEQ_LEN):
                w_idx = seq_start + t
                recon_errors[w_idx] += mse[b_idx, t]
                recon_counts[w_idx] += 1

recon_counts = np.maximum(recon_counts, 1)
lstm_scores_test = recon_errors / recon_counts
print(f"done  range: [{lstm_scores_test.min():.4f}, {lstm_scores_test.max():.4f}]")

torch.save(lstm_ae.state_dict(), f'{PROJECT_DIR}/models/lstm_autoencoder.pt')


# Ensemble: rank averaging across all four detectors
def rank_average(*score_arrays):
    ranks = [stats.rankdata(s) / len(s) for s in score_arrays]
    return np.mean(ranks, axis=0)

ensemble_scores = rank_average(iso_scores_test, lof_scores_test, ocsvm_scores_test, lstm_scores_test)
print(f"Ensemble range: [{ensemble_scores.min():.4f}, {ensemble_scores.max():.4f}]")


# ============================================================
# SECTION 12: EVALUATE AGAINST FLARE CATALOG
# ============================================================

def evaluate_detector(scores, feature_df_test, flare_catalog, pre_flare_hours=24,
                      name="Detector", min_consec_hours=3):
    """
    Evaluate an anomaly detector against the flare catalog.

    Threshold: Youden's J statistic (maximises tpr - fpr on the ROC curve),
               falling back to 95th percentile when only one class is present.

    Temporal consistency: an alert is only raised when the smoothed score
    exceeds the threshold for at least `min_consec_hours` consecutive hours,
    cutting spurious single-window spikes.

    Metrics: precision, window_recall, and F1 are all computed at window level
    so they are mutually consistent. event_recall counts how many distinct
    flares had at least one alert in their precursor window.
    """
    test_starts   = feature_df_test['window_start'].values
    test_ends     = feature_df_test['window_end'].values
    window_labels = np.zeros(len(scores))

    for _, flare in flare_catalog.iterrows():
        flare_start = pd.Timestamp(flare['start_time'])
        pre_start   = flare_start - pd.Timedelta(hours=pre_flare_hours)
        in_zone = (
            (test_starts >= np.datetime64(pre_start)) &
            (test_ends   <= np.datetime64(flare_start))
        )
        window_labels[in_zone] = 1

    roc_auc = roc_auc_score(window_labels, scores) if len(np.unique(window_labels)) > 1 else 0.0

    # --- Threshold: Youden's J ---
    if len(np.unique(window_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(window_labels, scores)
        best_idx  = np.argmax(tpr - fpr)
        threshold = thresholds[best_idx]
    else:
        threshold = np.percentile(scores, 95)

    # --- Temporal consistency filtering ---
    # Smooth with a rolling median then require min_consec_hours consecutive
    # windows above threshold before raising an alert.
    smoothed = pd.Series(scores).rolling(min_consec_hours, min_periods=1).median().values
    raw_preds = (smoothed >= threshold).astype(int)

    # Require a run of min_consec_hours consecutive 1s
    predictions = np.zeros(len(raw_preds), dtype=int)
    consec = 0
    for i, p in enumerate(raw_preds):
        if p == 1:
            consec += 1
            if consec >= min_consec_hours:
                predictions[i] = 1
        else:
            consec = 0

    # --- Event-level recall (how many distinct flares had ≥1 alert) ---
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

    n_flares      = len(flare_catalog)
    event_recall  = flares_detected / n_flares if n_flares > 0 else 0

    # --- Window-level metrics (all at same granularity → F1 is interpretable) ---
    tp             = ((predictions == 1) & (window_labels == 1)).sum()
    fp             = ((predictions == 1) & (window_labels == 0)).sum()
    fn             = ((predictions == 0) & (window_labels == 1)).sum()
    precision      = tp / (tp + fp) if (tp + fp) > 0 else 0
    window_recall  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1             = (2 * precision * window_recall / (precision + window_recall)
                      if (precision + window_recall) > 0 else 0)
    avg_lead       = np.mean(lead_times) if lead_times else 0

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Flares detected (event recall): {flares_detected} / {n_flares}  ({event_recall:.4f})")
    print(f"  Window Recall:   {window_recall:.4f}")
    print(f"  Precision:       {precision:.4f}")
    print(f"  F1 (window):     {f1:.4f}")
    print(f"  ROC AUC:         {roc_auc:.4f}")
    print(f"  Threshold:       {threshold:.4f}  (Youden's J)")
    print(f"  Avg Lead Time:   {avg_lead:.1f} hours")

    return dict(detector=name, event_recall=event_recall, window_recall=window_recall,
                precision=precision, f1=f1, roc_auc=roc_auc,
                avg_lead_time_hours=avg_lead,
                flares_detected=flares_detected, total_flares=n_flares)


feature_df_test = feature_df.loc[test_mask].copy()
test_flares = mx_flares[pd.to_datetime(mx_flares['start_time']) >= pd.Timestamp('2020-01-01')]

# --- Comparison baselines ---
np.random.seed(42)
random_scores = np.random.rand(len(ensemble_scores))

# Rate-of-change baseline: max absolute difference of xrsb_mean between
# adjacent windows, aligned to the test set index.
test_xrsb_mean = feature_df_test['xrsb_mean'].values
roc_baseline_scores = np.abs(np.diff(test_xrsb_mean, prepend=test_xrsb_mean[0]))

results = []
results.append(evaluate_detector(random_scores,      feature_df_test, test_flares, name="Random Baseline"))
results.append(evaluate_detector(roc_baseline_scores, feature_df_test, test_flares, name="Rate-of-Change Baseline"))
results.append(evaluate_detector(iso_scores_test,    feature_df_test, test_flares, name="Isolation Forest"))
results.append(evaluate_detector(lof_scores_test,    feature_df_test, test_flares, name="Local Outlier Factor"))
results.append(evaluate_detector(ocsvm_scores_test,  feature_df_test, test_flares, name="One-Class SVM"))
results.append(evaluate_detector(lstm_scores_test,   feature_df_test, test_flares, name="LSTM Autoencoder"))
results.append(evaluate_detector(ensemble_scores,    feature_df_test, test_flares, name="Ensemble (Rank Average)"))

results_df = pd.DataFrame(results)
print(f"\n{'='*70}")
print(f"  RESULTS SUMMARY")
print(f"{'='*70}")
print(results_df[['detector', 'event_recall', 'window_recall', 'precision',
                   'f1', 'roc_auc', 'avg_lead_time_hours']].to_string(index=False))
results_df.to_csv(f'{RESULTS_DIR}/baseline_results.csv', index=False)


# ============================================================
# SECTION 13: M-CLASS vs X-CLASS BREAKDOWN (all detectors)
# ============================================================
m_flares_test = test_flares[test_flares['class_letter'] == 'M']
x_flares_test = test_flares[test_flares['class_letter'] == 'X']

print(f"\nTest flares: {len(m_flares_test)} M-class, {len(x_flares_test)} X-class")

mx_results = []
for scores, det_name in [
    (iso_scores_test,    "Isolation Forest"),
    (lof_scores_test,    "Local Outlier Factor"),
    (ocsvm_scores_test,  "One-Class SVM"),
    (lstm_scores_test,   "LSTM Autoencoder"),
    (ensemble_scores,    "Ensemble"),
]:
    for cls_label, subset in [("M", m_flares_test), ("X", x_flares_test)]:
        if len(subset) == 0:
            continue
        r = evaluate_detector(scores, feature_df_test, subset,
                              name=f"{det_name} — {cls_label}-class")
        mx_results.append(r)

mx_df = pd.DataFrame(mx_results)
print(f"\n{'='*70}")
print(f"  M vs X CLASS BREAKDOWN")
print(f"{'='*70}")
print(mx_df[['detector', 'event_recall', 'window_recall', 'precision',
             'f1', 'roc_auc', 'avg_lead_time_hours']].to_string(index=False))
mx_df.to_csv(f'{RESULTS_DIR}/mx_class_results.csv', index=False)


# ============================================================
# SECTION 14: ANOMALY SCORE TIMELINE PLOT
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

test_times  = pd.to_datetime(feature_df_test['window_start'].values)
# Use Youden's J threshold derived from ensemble scores + window labels
_fpr, _tpr, _thresh = roc_curve(window_labels, ensemble_scores)
youden_threshold = _thresh[np.argmax(_tpr - _fpr)]

axes[0].plot(test_times, ensemble_scores, linewidth=0.3, color='steelblue', alpha=0.7)
axes[0].axhline(youden_threshold, color='red', linestyle='--', alpha=0.7,
                linewidth=1, label="Youden's J threshold")
for ft in pd.to_datetime(test_flares['start_time']):
    axes[0].axvline(ft, color='orange', alpha=0.15, linewidth=0.5)
axes[0].set_ylabel('Ensemble Anomaly Score')
axes[0].set_title('Anomaly Scores vs Actual M/X Flares (orange lines)')
axes[0].legend(loc='upper right')

test_flux = df.loc[df.index >= '2020-01-01', 'xrsb']
axes[1].semilogy(test_flux.index, test_flux, linewidth=0.15, color='navy', alpha=0.6)
axes[1].axhline(1e-5, color='orange', linestyle='--', alpha=0.4, label='M-class')
axes[1].axhline(1e-4, color='red',    linestyle='--', alpha=0.4, label='X-class')
axes[1].set_ylabel('XRS-B Flux (W/m²)')
axes[1].set_xlabel('Date')
axes[1].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/anomaly_scores_timeline.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# SECTION 15: SCORE DISTRIBUTIONS — QUIET vs PRE-FLARE
# ============================================================
test_starts   = feature_df_test['window_start'].values
window_labels = np.zeros(len(ensemble_scores))

for _, flare in test_flares.iterrows():
    flare_start = pd.Timestamp(flare['start_time'])
    pre_start   = flare_start - pd.Timedelta(hours=24)
    in_zone = (
        (test_starts >= np.datetime64(pre_start)) &
        (test_starts <= np.datetime64(flare_start))
    )
    window_labels[in_zone] = 1

quiet_scores    = ensemble_scores[window_labels == 0]
preflare_scores = ensemble_scores[window_labels == 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(quiet_scores,    bins=50, alpha=0.6, density=True, color='steelblue', label=f'Quiet ({len(quiet_scores):,})')
axes[0].hist(preflare_scores, bins=50, alpha=0.6, density=True, color='coral',     label=f'Pre-flare ({len(preflare_scores):,})')
axes[0].axvline(youden_threshold, color='red', linestyle='--', alpha=0.7, label="Youden's J threshold")
axes[0].set_xlabel('Ensemble Anomaly Score')
axes[0].set_ylabel('Density')
axes[0].set_title('Score Distributions: Quiet vs Pre-Flare')
axes[0].legend()

bp = axes[1].boxplot([quiet_scores, preflare_scores], tick_labels=['Quiet', 'Pre-Flare'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_ylabel('Ensemble Anomaly Score')
axes[1].set_title('Score Comparison')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/score_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Quiet:     median={np.median(quiet_scores):.4f}, mean={np.mean(quiet_scores):.4f}")
print(f"Pre-flare: median={np.median(preflare_scores):.4f}, mean={np.mean(preflare_scores):.4f}")


# ============================================================
# SECTION 16: SPOT-CHECK — FIRST TRUE POSITIVE
# ============================================================
threshold   = youden_threshold
predictions = (ensemble_scores >= threshold).astype(int)
test_starts = feature_df_test['window_start'].values

found_tp = False
for _, flare in test_flares.iterrows():
    flare_start = pd.Timestamp(flare['start_time'])
    pre_start   = flare_start - pd.Timedelta(hours=24)
    in_zone = (
        (test_starts >= np.datetime64(pre_start)) &
        (test_starts <= np.datetime64(flare_start))
    )
    if predictions[in_zone].sum() > 0:
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

        score_mask  = (test_starts >= np.datetime64(plot_start)) & (test_starts <= np.datetime64(plot_end))
        score_times = pd.to_datetime(test_starts[score_mask])
        score_vals  = ensemble_scores[score_mask]

        axes[1].plot(score_times, score_vals, linewidth=1.5, color='steelblue', marker='o', markersize=2)
        axes[1].axhline(threshold, color='red', linestyle='--', alpha=0.7, label="Youden's J threshold")
        axes[1].axvline(flare_start, color='red',    linestyle='--', linewidth=2)
        axes[1].axvline(pre_start,   color='orange', linestyle=':',  linewidth=1.5)
        axes[1].set_ylabel('Ensemble Anomaly Score')
        axes[1].set_xlabel('Time')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/true_positive_example.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Flare: {flare['flare_class']} at {flare_start}")
        found_tp = True
        break

if not found_tp:
    print("No true positives found — check evaluation.")


# ============================================================
# FINAL SUMMARY
# ============================================================
ensemble_row = results_df[results_df['detector'] == 'Ensemble (Rank Average)'].iloc[0]
print(f"\n{'='*60}")
print(f"  PIPELINE — COMPLETE")
print(f"{'='*60}")
print(f"  Data:      GOES-16 XRS, Feb 2017 – Dec 2024")
print(f"  Rows:      {len(df):,} (1-minute cadence)")
print(f"  Flares:    {len(mx_flares)} M/X class")
print(f"  Windows:   {len(feature_df):,} ({len(feature_cols)} features each)")
print(f"  Train:     {train_clean.sum():,} clean quiet windows")
print(f"  Test:      {test_mask.sum():,} windows")
print(f"")
print(f"  Detectors: IF, LOF, OCSVM, LSTM Autoencoder → rank-averaged ensemble")
print(f"  Threshold: Youden's J  |  Temporal filter: {3}-hour consistency")
print(f"")
print(f"  RESULTS (Ensemble):")
print(f"    Event Recall:   {ensemble_row['event_recall']:.3f}")
print(f"    Window Recall:  {ensemble_row['window_recall']:.3f}")
print(f"    Precision:      {ensemble_row['precision']:.3f}")
print(f"    F1 (window):    {ensemble_row['f1']:.3f}")
print(f"    ROC AUC:        {ensemble_row['roc_auc']:.3f}")
print(f"    Lead Time:      {ensemble_row['avg_lead_time_hours']:.1f} hours")
print(f"")
print(f"  FILES SAVED:")
print(f"    data/processed/goes_xrs_2017_2024.csv")
print(f"    data/features/baseline_features.csv")
print(f"    data/flare_catalog/mx_flares_2017_2024.csv")
print(f"    models/lstm_autoencoder.pt")
print(f"    results/baseline_results.csv")
print(f"    results/mx_class_results.csv")
print(f"    results/full_timeseries.png")
print(f"    results/flare_distribution.png")
print(f"    results/anomaly_scores_timeline.png")
print(f"    results/score_distributions.png")
print(f"    results/true_positive_example.png")
print(f"{'='*60}")
