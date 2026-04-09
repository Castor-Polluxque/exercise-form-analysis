"""
classify_exercise.py
---------------------
Classifies repetition-based exercises from multi-device accelerometer data.

Directory structure expected:
    <data_dir>/
        <exercise_name>/          e.g. situp, pushup, squat
            <recording_id>/       e.g. 1, 2, 3
                <device_files>    CSV files with columns:
                                  absolute_timestamp, accel_x, accel_y, accel_z

Device type is inferred from filename keywords:
    "bose" | "headphone"  → headphones
    "garmin" | "watch"    → watch
    "samsung" | "phone"   → phone

Rep segmentation uses the same pipeline as plot_multi_accel.py:
    1. Low-pass filter the vector magnitude
    2. Estimate rep period from the autocorrelation of the primary device
    3. Detect valleys with that period as the minimum separation
    4. Slice valley-to-valley segments on all devices

Features are computed per rep per device (orientation-invariant where possible)
and concatenated into a single feature vector. One row per rep.

Model: Random Forest (recording-level train/test split to avoid data leakage).

Usage:
    python classify_exercise.py <data_dir> [--primary-device watch]
                                            [--lowpass-hz 5.0]
                                            [--prominence 0.5]
                                            [--test-size 0.2]
                                            [--n-estimators 200]
                                            [--save-model model.joblib]
                                            [--save-features features.csv]
                                            [--save-plot importance.png]
"""

import argparse
import csv
import datetime
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GARMIN_EPOCH_OFFSET = 631065600   # seconds between 1989-12-31 and 1970-01-01

DEVICE_KEYWORDS = {
    'headphones': ['bose', 'headphone'],
    'watch':      ['garmin', 'watch'],
    'phone':      ['samsung', 'phone'],
}

# ---------------------------------------------------------------------------
# Timestamp detection  (identical to plot_multi_accel.py)
# ---------------------------------------------------------------------------

def detect_and_convert_timestamps(raw_timestamps):
    sample = raw_timestamps[len(raw_timestamps) // 2]
    if sample >= 1e15:
        return [t / 1e9 for t in raw_timestamps]
    elif sample >= 1e11:
        return [t / 1e3 for t in raw_timestamps]
    else:
        converted = list(raw_timestamps)
        if converted[len(converted) // 2] < 1262304000:
            converted = [t + GARMIN_EPOCH_OFFSET for t in raw_timestamps]
        return converted


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_device_csv(filepath):
    """Return (timestamps_unix_s, xs, ys, zs) sorted by timestamp."""
    raw_ts, xs, ys, zs = [], [], [], []
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_ts.append(float(row['absolute_timestamp']))
            xs.append(float(row['accel_x']))
            ys.append(float(row['accel_y']))
            zs.append(float(row['accel_z']))
    if not raw_ts:
        raise ValueError(f"No data in {filepath}")
    timestamps = detect_and_convert_timestamps(raw_ts)
    order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    return (
        np.array([timestamps[i] for i in order]),
        np.array([xs[i]         for i in order]),
        np.array([ys[i]         for i in order]),
        np.array([zs[i]         for i in order]),
    )


def infer_device_type(filename):
    """Return a canonical device key or None if unrecognised."""
    name = filename.lower()
    for device, keywords in DEVICE_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return device
    return None


def load_recording(recording_dir):
    """
    Load all recognised device files in a recording directory.

    Returns
    -------
    dict  keyed by device type → {'ts', 'xs', 'ys', 'zs'}
    """
    devices = {}
    for fpath in sorted(Path(recording_dir).iterdir()):
        if fpath.suffix.lower() != '.csv':
            continue
        dtype = infer_device_type(fpath.name)
        if dtype is None:
            print(f"    [skip] unrecognised device: {fpath.name}")
            continue
        print('fpath', fpath)
        ts, xs, ys, zs = load_device_csv(fpath)
        devices[dtype] = {'ts': ts, 'xs': xs, 'ys': ys, 'zs': zs, 'label': dtype}
    return devices


# ---------------------------------------------------------------------------
# Signal processing  (adapted from plot_multi_accel.py)
# ---------------------------------------------------------------------------

def lowpass_filter(signal, fs, cutoff_hz=5.0):
    nyq    = fs / 2.0
    cutoff = min(cutoff_hz, nyq * 0.9)
    b, a   = butter(2, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)


def compute_fs(times):
    dt = float(np.median(np.diff(times)))
    return 1.0 / dt if dt > 0 else 100.0


def estimate_rep_period_acf(mag_f, fs, min_period_s=0.3, max_period_s=10.0):
    """Return dominant rep period in seconds via normalised ACF."""
    x     = mag_f - mag_f.mean()
    n     = len(x)
    fft_x = np.fft.rfft(x, n=2 * n)
    acf   = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    acf   = acf / (acf[0] + 1e-12)

    min_lag = max(1, int(min_period_s * fs))
    max_lag = min(n - 1, int(max_period_s * fs))

    if min_lag >= max_lag:
        return max_period_s / 2.0

    peaks, props = find_peaks(acf[min_lag:max_lag], prominence=0.05)
    if len(peaks) == 0:
        return max_period_s / 2.0

    best     = peaks[np.argmax(props['prominences'])]
    period_s = (best + min_lag) / fs
    return period_s


def detect_valleys(times, xs, ys, zs, lowpass_hz=5.0,
                   min_separation_s=None, prominence_factor=0.5):
    """
    Return (valley_idx, mag_f) for the given device signal.
    min_separation_s=None → derived from ACF.
    """
    fs    = compute_fs(times)
    mag   = np.sqrt(xs**2 + ys**2 + zs**2)
    mag_f = lowpass_filter(mag, fs, lowpass_hz)

    if min_separation_s is None:
        period_s         = estimate_rep_period_acf(mag_f, fs)
        min_separation_s = period_s / 2.0

    dist    = max(1, int(min_separation_s * fs))
    iqr     = float(np.percentile(mag_f, 75) - np.percentile(mag_f, 25))
    prom    = max(iqr * prominence_factor, 1e-6)
    vidx, _ = find_peaks(-mag_f, distance=dist, prominence=prom)
    return vidx, mag_f, min_separation_s


def segment_reps(valley_idx, min_dur_factor=0.5):
    """Return list of (start_idx, end_idx) valley-to-valley segments."""
    if len(valley_idx) < 2:
        return []
    durs       = np.diff(valley_idx)
    median_dur = float(np.median(durs))
    min_dur    = max(1, int(median_dur * min_dur_factor))
    return [
        (int(valley_idx[i]), int(valley_idx[i + 1]))
        for i in range(len(valley_idx) - 1)
        if valley_idx[i + 1] - valley_idx[i] >= min_dur
    ]


def trim_to_window(ts, xs, ys, zs, t_start, t_end):
    mask = (ts >= t_start) & (ts <= t_end)
    return ts[mask], xs[mask], ys[mask], zs[mask]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def dominant_frequency(signal, fs):
    """Frequency (Hz) of the highest-amplitude FFT bin above 0 Hz."""
    n      = len(signal)
    if n < 4:
        return 0.0
    fft    = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
    fft[0] = 0.0   # zero DC
    return float(freqs[np.argmax(fft)])


def top3_power_ratio(signal):
    """Fraction of total FFT power in the top 3 magnitude bins."""
    fft   = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    total = fft.sum()
    if total < 1e-12:
        return 0.0
    top3  = np.sort(fft)[-3:].sum()
    return float(top3 / total)


def zero_crossing_rate(signal):
    """Number of zero crossings per sample."""
    centred = signal - signal.mean()
    zc      = np.sum(np.diff(np.sign(centred)) != 0)
    return float(zc) / max(len(signal) - 1, 1)


def safe_pearson(a, b):
    """Pearson r, returns 0 if either signal has zero variance."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_device_features(mag_f_seg, xs_seg, ys_seg, zs_seg,
                              fs, rep_duration_s):
    """
    Compute all features for one rep on one device.

    Returns a flat dict of {feature_name: value}.
    """
    feats = {}

    # --- Magnitude features (orientation-invariant) ---
    mag = mag_f_seg   # already filtered
    feats['mag_mean']     = float(np.mean(mag))
    feats['mag_std']      = float(np.std(mag))
    feats['mag_min']      = float(np.min(mag))
    feats['mag_max']      = float(np.max(mag))
    feats['mag_range']    = feats['mag_max'] - feats['mag_min']
    feats['mag_rms']      = float(np.sqrt(np.mean(mag**2)))
    feats['mag_skew']     = float(_safe_skewness(mag))
    feats['mag_kurt']     = float(_safe_kurtosis(mag))
    feats['mag_dom_freq'] = dominant_frequency(mag, fs)
    feats['mag_top3_pwr'] = top3_power_ratio(mag)

    # Fraction of rep spent above the magnitude mean (phase asymmetry)
    above_mean = float(np.sum(mag > feats['mag_mean'])) / max(len(mag), 1)
    feats['mag_frac_above_mean'] = above_mean

    # --- Per-axis features ---
    for axis_name, axis_sig in [('x', xs_seg), ('y', ys_seg), ('z', zs_seg)]:
        feats[f'{axis_name}_mean'] = float(np.mean(axis_sig))
        feats[f'{axis_name}_std']  = float(np.std(axis_sig))
        feats[f'{axis_name}_zcr']  = zero_crossing_rate(axis_sig)

    # --- Cross-axis correlations ---
    feats['corr_xy'] = safe_pearson(xs_seg, ys_seg)
    feats['corr_xz'] = safe_pearson(xs_seg, zs_seg)
    feats['corr_yz'] = safe_pearson(ys_seg, zs_seg)

    # --- Rep timing ---
    feats['rep_duration_s'] = float(rep_duration_s)

    return feats


def _safe_skewness(x):
    mu, sd = np.mean(x), np.std(x)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 3))


def _safe_kurtosis(x):
    mu, sd = np.mean(x), np.std(x)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 4) - 3.0)


def features_for_recording(devices, primary_device='watch',
                             lowpass_hz=5.0, prominence_factor=0.5):
    """
    Segment reps from the primary device and extract features from all devices.

    Returns
    -------
    list of dicts, one per rep.  Each dict contains prefixed feature names
    like 'watch_mag_mean', 'headphones_corr_xy', etc., plus 'rep_duration_s'.
    Returns empty list if segmentation fails.
    """
    # ------------------------------------------------------------------
    # Trim all devices to the overlapping time window
    # ------------------------------------------------------------------
    t_start = max(d['ts'][0]  for d in devices.values())
    t_end   = min(d['ts'][-1] for d in devices.values())
    if t_start >= t_end:
        print("    [skip] no overlapping time window")
        return []

    trimmed = {}
    for name, d in devices.items():
        ts, xs, ys, zs = trim_to_window(
            d['ts'], d['xs'], d['ys'], d['zs'], t_start, t_end)
        if len(ts) < 10:
            print(f"    [skip] {name} has too few samples after trim")
            return []
        times = ts - ts[0]
        trimmed[name] = {'times': times, 'xs': xs, 'ys': ys, 'zs': zs}

    # ------------------------------------------------------------------
    # Segment on primary device
    # ------------------------------------------------------------------
    if primary_device not in trimmed:
        # Fall back to first available device
        primary_device = next(iter(trimmed))
        print(f"    [warn] primary device not found, using {primary_device}")

    pd      = trimmed[primary_device]
    fs_p    = compute_fs(pd['times'])
    mag_p   = np.sqrt(pd['xs']**2 + pd['ys']**2 + pd['zs']**2)
    mag_p_f = lowpass_filter(mag_p, fs_p, lowpass_hz)

    valley_idx, _, min_sep = detect_valleys(
        pd['times'], pd['xs'], pd['ys'], pd['zs'],
        lowpass_hz=lowpass_hz,
        prominence_factor=prominence_factor,
    )
    segments = segment_reps(valley_idx)

    if len(segments) < 2:
        print(f"    [skip] only {len(segments)} rep(s) detected")
        return []

    print(f"    {len(segments)} reps detected "
          f"(min_sep={min_sep:.2f}s, primary={primary_device})")

    # ------------------------------------------------------------------
    # Extract features per rep per device
    # ------------------------------------------------------------------
    all_rep_features = []

    for seg_i, (s_idx, e_idx) in enumerate(segments):
        # Map primary-device sample indices to time boundaries
        t_rep_start = pd['times'][s_idx]
        t_rep_end   = pd['times'][min(e_idx, len(pd['times']) - 1)]
        rep_dur     = float(t_rep_end - t_rep_start)

        if rep_dur <= 0:
            continue

        rep_feats = {'rep_duration_s': rep_dur}

        for dev_name, ddata in trimmed.items():
            # Slice this device using the primary-device time boundaries
            mask = (ddata['times'] >= t_rep_start) & \
                   (ddata['times'] <= t_rep_end)
            xs_s = ddata['xs'][mask]
            ys_s = ddata['ys'][mask]
            zs_s = ddata['zs'][mask]

            if len(xs_s) < 4:
                # Not enough samples from this device for this rep window
                # Fill with NaN so the row is still usable after imputation
                stub = {f'{dev_name}_{k}': np.nan
                        for k in ['mag_mean', 'mag_std', 'mag_min', 'mag_max',
                                  'mag_range', 'mag_rms', 'mag_skew',
                                  'mag_kurt', 'mag_dom_freq', 'mag_top3_pwr',
                                  'mag_frac_above_mean',
                                  'x_mean', 'x_std', 'x_zcr',
                                  'y_mean', 'y_std', 'y_zcr',
                                  'z_mean', 'z_std', 'z_zcr',
                                  'corr_xy', 'corr_xz', 'corr_yz']}
                rep_feats.update(stub)
                continue

            fs_d    = compute_fs(ddata['times'][mask])
            mag_s   = np.sqrt(xs_s**2 + ys_s**2 + zs_s**2)
            mag_s_f = lowpass_filter(mag_s, fs_d, lowpass_hz) \
                      if len(mag_s) > 9 else mag_s

            dev_feats = extract_device_features(
                mag_s_f, xs_s, ys_s, zs_s, fs_d, rep_dur)

            # Prefix feature names with device name; skip rep_duration_s
            # (already added once above, shared across devices)
            for k, v in dev_feats.items():
                if k != 'rep_duration_s':
                    rep_feats[f'{dev_name}_{k}'] = v

        all_rep_features.append(rep_feats)

    return all_rep_features


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(data_dir, primary_device='watch',
                  lowpass_hz=5.0, prominence_factor=0.5):
    """
    Walk data_dir/<exercise>/<recording_id>/ and build a flat feature table.

    Returns
    -------
    rows        : list of feature dicts
    labels      : list of exercise name strings (one per rep)
    group_ids   : list of recording ID strings (one per rep, for grouped CV)
    """
    data_dir = Path(data_dir)
    rows, labels, group_ids = [], [], []

    exercise_dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir())

    for ex_dir in exercise_dirs:
        exercise = ex_dir.name
        rec_dirs = sorted(
            p for p in ex_dir.iterdir() if p.is_dir())

        if not rec_dirs:
            print(f"  [{exercise}] no recording subdirectories found, skipping")
            continue

        print(f"\n[{exercise}]")
        for rec_dir in rec_dirs:
            rec_id = f"{exercise}/{rec_dir.name}"
            print(f"  Recording: {rec_dir.name}")

            devices = load_recording(rec_dir)
            if len(devices) == 0:
                print("    [skip] no recognised device files")
                continue

            rep_feats = features_for_recording(
                devices,
                primary_device=primary_device,
                lowpass_hz=lowpass_hz,
                prominence_factor=prominence_factor,
            )

            for rf in rep_feats:
                rows.append(rf)
                labels.append(exercise)
                group_ids.append(rec_id)

    return rows, labels, group_ids


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(rows, labels, group_ids,
                       test_size=0.2, n_estimators=200, random_state=42):
    """
    Train a Random Forest with a recording-level grouped train/test split.

    Returns
    -------
    clf          : fitted RandomForestClassifier
    feature_names: list of feature name strings
    X_test       : test feature matrix
    y_test       : true labels for test set
    y_pred       : predicted labels for test set
    """
    import pandas as pd
    from sklearn.impute import SimpleImputer

    # Build DataFrame to align columns across all reps
    df = pd.DataFrame(rows).fillna(method='ffill').fillna(0.0)
    feature_names = list(df.columns)

    X = df.values.astype(float)
    y = np.array(labels)
    g = np.array(group_ids)

    # Impute any remaining NaNs with column median
    imputer = SimpleImputer(strategy='median')
    X       = imputer.fit_transform(X)

    # Recording-level split: entire recordings go to either train or test
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=g))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nTrain: {len(X_train)} reps from "
          f"{len(set(g[train_idx]))} recordings")
    print(f"Test:  {len(X_test)} reps from "
          f"{len(set(g[test_idx]))} recordings")
    print(f"Train classes: {sorted(set(y_train))}")
    print(f"Test  classes: {sorted(set(y_test))}")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, feature_names, imputer, X_test, y_test, y_pred


# ---------------------------------------------------------------------------
# Reporting and plotting
# ---------------------------------------------------------------------------

def print_report(y_test, y_pred):
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred))

    print("CONFUSION MATRIX")
    classes = sorted(set(y_test) | set(y_pred))
    cm      = confusion_matrix(y_test, y_pred, labels=classes)
    col_w   = max(len(c) for c in classes) + 2
    header  = " " * col_w + "".join(f"{c:>{col_w}}" for c in classes)
    print(header)
    for true_c, row in zip(classes, cm):
        print(f"{true_c:<{col_w}}" +
              "".join(f"{v:>{col_w}}" for v in row))
    print()


def plot_feature_importance(clf, feature_names, top_n=30, save_path=None):
    importances = clf.feature_importances_
    idx         = np.argsort(importances)[-top_n:][::-1]
    top_names   = [feature_names[i] for i in idx]
    top_vals    = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    bars = ax.barh(range(top_n), top_vals[::-1],
                   color='#3498db', edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean decrease in impurity (feature importance)", fontsize=10)
    ax.set_title(f"Top {top_n} features — Random Forest", fontsize=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.grid(True, axis='x', which='minor', linestyle=':', alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Feature importance plot saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train an exercise classifier from multi-device '
                    'accelerometer data.')
    parser.add_argument('data_dir',
                        help='Root data directory. Structure: '
                             '<data_dir>/<exercise>/<recording_id>/<csvs>')
    parser.add_argument('--primary-device', default='watch',
                        choices=['watch', 'headphones', 'phone'],
                        help='Device used for rep segmentation (default: watch)')
    parser.add_argument('--lowpass-hz', type=float, default=5.0,
                        help='Low-pass filter cutoff in Hz (default: 5.0)')
    parser.add_argument('--prominence', type=float, default=0.5,
                        help='Valley prominence factor × IQR (default: 0.5)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of recordings held out for test '
                             '(default: 0.2)')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of trees in the Random Forest '
                             '(default: 200)')
    parser.add_argument('--save-model', metavar='PATH', default=None,
                        help='Save the trained model to this path '
                             '(joblib format)')
    parser.add_argument('--save-features', metavar='PATH', default=None,
                        help='Save the full feature matrix to this CSV path')
    parser.add_argument('--save-plot', metavar='PATH', default=None,
                        help='Save the feature importance plot to this path')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    print("Building dataset ...")
    rows, labels, group_ids = build_dataset(
        args.data_dir,
        primary_device=args.primary_device,
        lowpass_hz=args.lowpass_hz,
        prominence_factor=args.prominence,
    )

    if len(rows) == 0:
        print("\nNo reps extracted. Check data directory structure and "
              "device file naming.")
        return

    print(f"\nTotal reps extracted: {len(rows)}")
    from collections import Counter
    for ex, cnt in sorted(Counter(labels).items()):
        print(f"  {ex}: {cnt} reps")

    # ------------------------------------------------------------------
    # Optionally save feature matrix
    # ------------------------------------------------------------------
    if args.save_features:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.insert(0, 'exercise', labels)
        df.insert(1, 'recording', group_ids)
        df.to_csv(args.save_features, index=False)
        print(f"\nFeature matrix saved: {args.save_features}")

    # ------------------------------------------------------------------
    # Train and evaluate
    # ------------------------------------------------------------------
    n_classes    = len(set(labels))
    n_recordings = len(set(group_ids))
    if n_classes < 2:
        print("\nNeed at least 2 exercise classes to train a classifier.")
        return
    if n_recordings < 2:
        print("\nNeed at least 2 recordings to perform a train/test split.")
        return

    clf, feature_names, imputer, X_test, y_test, y_pred = train_and_evaluate(
        rows, labels, group_ids,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
    )

    print_report(y_test, y_pred)

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    if args.save_model:
        joblib.dump({'model': clf, 'imputer': imputer,
                     'feature_names': feature_names}, args.save_model)
        print(f"Model saved: {args.save_model}")

    # ------------------------------------------------------------------
    # Feature importance plot
    # ------------------------------------------------------------------
    plot_feature_importance(
        clf, feature_names,
        top_n=min(30, len(feature_names)),
        save_path=args.save_plot,
    )


if __name__ == '__main__':
    main()
