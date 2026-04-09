"""
plot_multi_accel.py
--------------------
Plots X, Y, Z accelerometer data from 3 devices on a single figure,
synchronized to their overlapping time window. Performs peak and valley
detection on the vector magnitude of each device and annotates them on
every axis subplot:
  ▼  (downward triangle) = peak   (high-acceleration moment)
  ▲  (upward triangle)   = valley (low-acceleration moment)

Rep segmentation is performed on a single primary device (default: device 0,
typically the wrist) using valley-to-valley boundaries. The minimum separation
between valleys is estimated automatically from the autocorrelation of the
filtered magnitude signal, so no exercise-specific tuning is required. A
manual override (--min-separation) is available for edge cases. The first N
reps (default: 5) are used as a good-form template. Every rep is scored
against that template using DTW on the raw vector magnitude of each device
independently. Per-device scores are normalised to a common scale and then
combined via a weighted sum (equal weights by default, overridable with
--weights). Reps whose combined score exceeds the threshold are flagged as
anomalous and highlighted in red.

Each device's CSV must have at minimum these columns:
    absolute_timestamp, accel_x, accel_y, accel_z

Timestamp units and epochs are detected automatically:
  - Seconds      (~1e9):  treated as Unix epoch seconds
  - Milliseconds (~1e12): divided by 1e3
  - Nanoseconds  (~1e18): divided by 1e9
  - Garmin epoch (~1e9 but resolves to pre-2010): offset by +631065600 s

Usage:
    python plot_multi_accel.py <file1.csv> <file2.csv> <file3.csv> \\
        --labels "Bose" "Garmin" "Samsung" \\
        --primary 1 \\
        --template-reps 5 \\
        --anomaly-threshold 2.0 \\
        --weights 0.5 0.3 0.2 \\
        --save-png output.png
"""

import csv
import argparse
import datetime
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Garmin epoch offset (seconds between 1989-12-31 and 1970-01-01)
GARMIN_EPOCH_OFFSET = 631065600


# ---------------------------------------------------------------------------
# Timestamp detection
# ---------------------------------------------------------------------------

def detect_and_convert_timestamps(raw_timestamps):
    """
    Detect timestamp units/epoch and return as Unix epoch seconds (float).

    Detection order:
      1. ~1e18 → nanoseconds  (divide by 1e9)
      2. ~1e12 → milliseconds (divide by 1e3)
      3. ~1e9  → seconds; if resulting date is before 2010 assume Garmin epoch
    """
    sample = raw_timestamps[len(raw_timestamps) // 2]

    if sample >= 1e15:
        converted = [t / 1e9 for t in raw_timestamps]
        unit = "nanoseconds"
    elif sample >= 1e11:
        converted = [t / 1e3 for t in raw_timestamps]
        unit = "milliseconds"
    else:
        converted = list(raw_timestamps)
        unit = "seconds (Unix)"
        # Secondary check: if midpoint is before 2010, it's Garmin epoch
        if converted[len(converted) // 2] < 1262304000:  # 2010-01-01
            converted = [t + GARMIN_EPOCH_OFFSET for t in raw_timestamps]
            unit = "seconds (Garmin epoch → Unix)"

    mid = converted[len(converted) // 2]
    dt  = datetime.datetime.fromtimestamp(mid, tz=datetime.timezone.utc)
    print(f"    Unit detected: {unit}")
    print(f"    Midpoint time: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    return converted


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_device_csv(filepath):
    """
    Load a clean accelerometer CSV.
    Returns timestamps (Unix epoch seconds) and xs, ys, zs arrays,
    sorted by timestamp.
    """
    raw_ts, xs, ys, zs = [], [], [], []

    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_ts.append(float(row['absolute_timestamp']))
            xs.append(float(row['accel_x']))
            ys.append(float(row['accel_y']))
            zs.append(float(row['accel_z']))

    if not raw_ts:
        raise ValueError(f"No data rows found in {filepath}")

    timestamps = detect_and_convert_timestamps(raw_ts)

    # Sort by timestamp (some devices write out-of-order)
    order      = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    timestamps = [timestamps[i] for i in order]
    xs         = [xs[i] for i in order]
    ys         = [ys[i] for i in order]
    zs         = [zs[i] for i in order]

    return (np.array(timestamps), np.array(xs),
            np.array(ys),         np.array(zs))


def trim_to_window(timestamps, xs, ys, zs, t_start, t_end):
    """Keep only samples within [t_start, t_end]."""
    mask = (timestamps >= t_start) & (timestamps <= t_end)
    return timestamps[mask], xs[mask], ys[mask], zs[mask]


# ---------------------------------------------------------------------------
# Autocorrelation-based rep period estimation
# ---------------------------------------------------------------------------

def estimate_rep_period_acf(mag_f, fs,
                             min_period_s=0.3,
                             max_period_s=10.0):
    """
    Estimate the dominant rep period from the autocorrelation of the filtered
    magnitude signal.

    The full-signal ACF is computed, then the first peak beyond
    `min_period_s` is found. That lag is the estimated rep period. Searching
    beyond a minimum lag prevents the zero-lag trivial peak and any harmonics
    at very short lags from being picked up.

    Parameters
    ----------
    mag_f        : 1-D array, filtered magnitude signal
    fs           : sampling rate in Hz
    min_period_s : shortest plausible rep duration in seconds (default 0.3 s).
                   Prevents the zero-lag peak and sub-rep harmonics from being
                   selected. 0.3 s is intentionally very short so the function
                   works for fast exercises like jumping jacks.
    max_period_s : longest plausible rep duration in seconds (default 10 s).
                   Acts as a safety cap for very slow exercises or noisy signals.

    Returns
    -------
    period_s : float, estimated rep period in seconds.
               Falls back to max_period_s / 2 if no clear peak is found.
    """
    # Mean-centre before computing ACF so DC offset doesn't dominate
    x     = mag_f - mag_f.mean()
    n     = len(x)
    # Full normalised ACF via FFT (O(n log n) rather than O(n²))
    fft_x = np.fft.rfft(x, n=2 * n)
    acf   = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    acf   = acf / (acf[0] + 1e-12)   # normalise to 1 at lag 0

    min_lag = max(1, int(min_period_s * fs))
    max_lag = min(n - 1, int(max_period_s * fs))

    if min_lag >= max_lag:
        fallback = max_period_s / 2.0
        print(f"    ACF: search range empty, falling back to {fallback:.2f} s")
        return fallback

    search_region = acf[min_lag:max_lag]
    peaks, props  = find_peaks(search_region, prominence=0.05)

    if len(peaks) == 0:
        # No clear periodicity found; use a conservative fallback
        fallback = max_period_s / 2.0
        print(f"    ACF: no dominant peak found, falling back to {fallback:.2f} s")
        return fallback

    # Pick the highest-prominence peak (most clearly periodic)
    best      = peaks[np.argmax(props['prominences'])]
    lag_idx   = best + min_lag          # index back into the full ACF
    period_s  = lag_idx / fs
    print(f"    ACF: estimated rep period = {period_s:.2f} s  "
          f"(ACF peak at lag {lag_idx}, prominence "
          f"{props['prominences'][np.argmax(props['prominences'])]:.3f})")
    return period_s


# ---------------------------------------------------------------------------
# Peak / valley detection
# ---------------------------------------------------------------------------

def detect_peaks_valleys(times, xs, ys, zs,
                          lowpass_hz=5.0,
                          min_separation_s=None,
                          prominence_factor=0.5):
    """
    Detect peaks and valleys in the vector magnitude signal.

    Steps:
      1. Compute vector magnitude: mag = sqrt(x^2 + y^2 + z^2)
      2. Low-pass filter at `lowpass_hz` Hz to remove high-freq noise
      3. If `min_separation_s` is None, estimate the rep period via
         autocorrelation and use half that period as the minimum separation
         (peaks and valleys of the same type recur once per full period,
         so half the period is the tightest safe lower bound)
      4. Run scipy find_peaks on the filtered magnitude with:
           - minimum separation of `min_separation_s` seconds
           - minimum prominence of `prominence_factor` x signal IQR
      5. Run find_peaks on the inverted signal for valleys

    Parameters
    ----------
    times             : 1-D array of normalised time values (seconds)
    xs, ys, zs        : 1-D arrays of accelerometer axes
    lowpass_hz        : low-pass filter cutoff frequency
    min_separation_s  : minimum seconds between same-type events. When None
                        (default) the value is derived automatically from the
                        ACF of the filtered magnitude.
    prominence_factor : IQR multiplier for the peak prominence threshold

    Returns
    -------
    peak_times   : 1-D array of times of peaks
    valley_times : 1-D array of times of valleys
    valley_idx   : 1-D array of sample indices of valleys (for segmentation)
    mag_f        : filtered magnitude array (for DTW scoring)
    min_sep_used : the min_separation_s value actually used (for logging)
    """
    dt = float(np.median(np.diff(times)))
    fs = 1.0 / dt if dt > 0 else 100.0

    mag = np.sqrt(xs**2 + ys**2 + zs**2)

    # Low-pass filter
    nyq    = fs / 2.0
    cutoff = min(lowpass_hz, nyq * 0.9)   # guard against low-fs devices
    b, a   = butter(2, cutoff / nyq, btype='low')
    mag_f  = filtfilt(b, a, mag)

    # Derive minimum separation from ACF when not supplied
    if min_separation_s is None:
        period_s     = estimate_rep_period_acf(mag_f, fs)
        # Half the rep period: same-type events (valley→valley) are separated
        # by one full period, so half is the tightest safe lower bound.
        min_sep_used = period_s / 2.0
    else:
        min_sep_used = min_separation_s

    dist_samples = max(1, int(min_sep_used * fs))

    # Use IQR-based prominence so outlier spikes do not inflate the threshold
    iqr        = float(np.percentile(mag_f, 75) - np.percentile(mag_f, 25))
    prominence = max(iqr * prominence_factor, 1e-6)

    peak_idx,   _ = find_peaks( mag_f, distance=dist_samples,
                                 prominence=prominence)
    valley_idx, _ = find_peaks(-mag_f, distance=dist_samples,
                                 prominence=prominence)

    return (np.array(times)[peak_idx],
            np.array(times)[valley_idx],
            valley_idx,
            mag_f,
            min_sep_used)


# ---------------------------------------------------------------------------
# Rep segmentation
# ---------------------------------------------------------------------------

def segment_reps(valley_idx, n_samples, min_dur_factor=0.5, n_template=5):
    """
    Slice the signal into valley-to-valley rep segments.

    Uses the first `n_template` inter-valley durations to estimate a typical
    rep length, then rejects any segment shorter than `min_dur_factor` × that
    median duration. This guards against spurious extra valleys (e.g. a wobble
    mid-rep) splitting a single rep into two short fragments.

    Parameters
    ----------
    valley_idx      : array of sample indices of detected valleys
    n_samples       : total number of samples in the signal
    min_dur_factor  : minimum rep length as a fraction of median rep length
    n_template      : how many early reps to use for the duration estimate

    Returns
    -------
    segments : list of (start_idx, end_idx) tuples, one per rep.
               The slice signal[start_idx:end_idx] covers one rep.
    """
    if len(valley_idx) < 2:
        return []

    # Estimate typical rep duration from the first n_template inter-valley gaps
    early_durs = np.diff(valley_idx[:n_template + 1])
    median_dur = float(np.median(early_durs))
    min_dur    = max(1, int(median_dur * min_dur_factor))

    segments = []
    for i in range(len(valley_idx) - 1):
        start = int(valley_idx[i])
        end   = int(valley_idx[i + 1])
        if (end - start) >= min_dur:
            segments.append((start, end))

    return segments


# ---------------------------------------------------------------------------
# DTW
# ---------------------------------------------------------------------------

def dtw_distance(a, b):
    """
    Compute the DTW distance between two 1-D signals using raw amplitude values.

    The total accumulated cost is divided by the warping path length so that
    longer reps are not penalised purely for having more samples (length
    normalisation). No amplitude normalisation is applied: differences in
    peak height are a meaningful bad-form signal and should contribute to
    the score.

    Parameters
    ----------
    a, b : 1-D numpy arrays

    Returns
    -------
    float : length-normalised DTW distance
    """
    n, m = len(a), len(b)
    # Initialise cost matrix with infinity
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost    = abs(float(a[i - 1]) - float(b[j - 1]))
            D[i, j] = cost + min(D[i - 1, j],      # insertion
                                  D[i, j - 1],      # deletion
                                  D[i - 1, j - 1])  # match

    # Trace back path length for normalisation
    i, j = n, m
    path_len = 0
    while i > 0 or j > 0:
        path_len += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

    return D[n, m] / max(path_len, 1)


# ---------------------------------------------------------------------------
# Template building
# ---------------------------------------------------------------------------

def build_template(segments, mag_f, n_template=5):
    """
    Build a good-form template from the first `n_template` rep segments.

    Uses the medoid strategy: picks the single rep (among the template reps)
    that has the lowest total DTW distance to all others. This avoids
    averaging artifacts while still being representative of the group.

    Parameters
    ----------
    segments   : list of (start_idx, end_idx) tuples from segment_reps()
    mag_f      : filtered magnitude array
    n_template : number of early reps to treat as good-form examples

    Returns
    -------
    template : 1-D numpy array — the medoid rep's magnitude signal
    """
    n = min(n_template, len(segments))
    if n < 1:
        raise ValueError("Need at least 1 rep to build a template.")

    segs = [mag_f[s:e] for s, e in segments[:n]]

    if n == 1:
        return segs[0].copy()

    # Compute pairwise DTW distances among the template reps
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(segs[i], segs[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Medoid = rep with lowest sum of distances to all others
    medoid_idx = int(np.argmin(dist_matrix.sum(axis=1)))
    print(f"    Template medoid: rep {medoid_idx + 1} of {n} "
          f"(total DTW sum = {dist_matrix[medoid_idx].sum():.4f})")
    return segs[medoid_idx].copy()


# ---------------------------------------------------------------------------
# Rep scoring
# ---------------------------------------------------------------------------

def score_reps(segments, devices, templates, weights,
               n_template=5, threshold_sigma=2.0):
    """
    Score every rep against the good-form template for each device, then
    combine into a single weighted anomaly score.

    Per-device scores are computed on raw amplitude (preserving depth-of-rep
    information), then normalised to zero-mean / unit-variance across reps
    before weighting. Normalisation is done per device so that a device whose
    magnitude signal naturally lives on a larger scale (e.g. wrist vs.
    headphones) does not dominate the combined score simply due to scale.

    The anomaly threshold is person-adaptive: it is set at
    `threshold_sigma` standard deviations above the mean combined score of
    the template reps themselves.

    Parameters
    ----------
    segments        : list of (start_idx, end_idx) tuples
    devices         : list of device dicts (must contain 'mag_f' and 'label')
    templates       : list of 1-D arrays, one template per device,
                      from build_template()
    weights         : 1-D array of per-device weights (normalised to sum-to-1
                      internally)
    n_template      : number of template reps (used to compute threshold)
    threshold_sigma : how many sigma above template-rep mean marks an anomaly

    Returns
    -------
    per_device_scores : 2-D array, shape (n_devices, n_reps) - raw DTW
                        distances per device before normalisation
    combined_scores   : 1-D array, shape (n_reps,) - weighted combined score
    threshold         : scalar anomaly threshold on the combined score
    anomalous         : boolean array, True where combined score > threshold
    """
    n_reps    = len(segments)
    n_devices = len(devices)

    # --- Raw DTW scores: shape (n_devices, n_reps) ---
    raw = np.zeros((n_devices, n_reps))
    for d_i, (dev, tmpl) in enumerate(zip(devices, templates)):
        mag_f = dev['mag_f']
        for r_i, (s, e) in enumerate(segments):
            raw[d_i, r_i] = dtw_distance(mag_f[s:e], tmpl)

    # --- Per-device z-score normalisation across reps ---
    # Equalises scale across devices without discarding amplitude information
    # within a device: each device's scores are centred and scaled by their
    # own cross-rep distribution, not by their raw amplitude units.
    normed = np.zeros_like(raw)
    for d_i in range(n_devices):
        mu    = float(np.mean(raw[d_i]))
        sigma = float(np.std(raw[d_i]))
        sigma = max(sigma, 1e-6)
        normed[d_i] = (raw[d_i] - mu) / sigma

    # --- Weighted sum across devices ---
    w = np.array(weights, dtype=float)
    w = w / w.sum()                                       # normalise to sum-to-1
    combined = (normed * w[:, np.newaxis]).sum(axis=0)    # shape (n_reps,)

    # --- Person-adaptive threshold from template reps ---
    template_combined = combined[:min(n_template, n_reps)]
    t_mu    = float(np.mean(template_combined))
    t_sigma = float(np.std(template_combined))
    t_sigma = max(t_sigma, 1e-6)
    threshold = t_mu + threshold_sigma * t_sigma

    anomalous = combined > threshold
    return raw, combined, threshold, anomalous


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(devices, segments, per_device_scores, combined_scores,
             threshold, anomalous, weights, primary_label, save_path=None):
    """
    One subplot per device showing X/Y/Z lines, peak/valley markers, and
    rep shading. A final subplot shows a stacked bar chart of the per-device
    weighted DTW contribution to the combined anomaly score.

    Rep shading:
      green = good-form rep (combined score within threshold)
      red   = anomalous rep (combined score exceeds threshold)

    Peaks   -> inverted triangle (v) above the signal
    Valleys -> upright triangle  (^) below the signal
    All device subplots share the same x-axis.

    Parameters
    ----------
    devices           : list of device dicts (label, times, xs, ys, zs, ...)
    segments          : list of (start_idx, end_idx) rep boundary tuples
    per_device_scores : 2-D array (n_devices, n_reps) of raw DTW distances
    combined_scores   : 1-D array (n_reps,) of weighted combined scores
    threshold         : scalar anomaly threshold on combined_scores
    anomalous         : boolean array (n_reps,)
    weights           : 1-D array of normalised per-device weights
    primary_label     : label string of the primary (segmentation) device
    save_path         : if given, save PNG here instead of displaying
    """
    axis_colors  = {'X': '#e74c3c', 'Y': '#2ecc71', 'Z': '#3498db'}
    peak_color   = '#c0392b'
    valley_color = '#27ae60'

    # Distinct colours for device bars in the stacked chart
    device_bar_colors = [
        '#3498db', '#e67e22', '#9b59b6',
        '#1abc9c', '#e74c3c', '#f1c40f',
    ]

    n_subplots = len(devices) + 1   # device signal axes + stacked score chart
    fig, axes  = plt.subplots(n_subplots, 1,
                               figsize=(14, 4 * n_subplots),
                               sharex=False)
    device_axes = axes[:len(devices)]
    score_ax    = axes[-1]

    # Share x-axis across device signal subplots only
    for i in range(len(devices) - 1):
        device_axes[i].get_shared_x_axes().joined(
            device_axes[i], device_axes[i + 1])

    fig.suptitle("Multi-Device Accelerometer Comparison (synchronised)",
                 fontsize=14, fontweight='bold')

    # Primary device times used to map segment indices to time values
    primary_times = np.array(devices[0]['times'])   # device 0 is always primary

    def shade_reps(ax):
        """Draw per-rep background shading and rep-number labels."""
        for rep_i, (s, e) in enumerate(segments):
            t_s = primary_times[s]
            t_e = primary_times[min(e, len(primary_times) - 1)]
            color = '#e74c3c' if anomalous[rep_i] else '#2ecc71'
            ax.axvspan(t_s, t_e, color=color, alpha=0.13, zorder=0)
            ax.text((t_s + t_e) / 2, 1.0, str(rep_i + 1),
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7, color='#555555')

    # ------------------------------------------------------------------
    # Device signal subplots
    # ------------------------------------------------------------------
    for ax, dev in zip(device_axes, devices):
        times        = np.array(dev['times'])
        xs           = np.array(dev['xs'])
        ys           = np.array(dev['ys'])
        zs           = np.array(dev['zs'])
        peak_times   = dev['peak_times']
        valley_times = dev['valley_times']

        shade_reps(ax)

        ax.plot(times, xs, color=axis_colors['X'],
                linewidth=0.7, alpha=0.9, label='X')
        ax.plot(times, ys, color=axis_colors['Y'],
                linewidth=0.7, alpha=0.9, label='Y')
        ax.plot(times, zs, color=axis_colors['Z'],
                linewidth=0.7, alpha=0.9, label='Z')

        y_min = min(xs.min(), ys.min(), zs.min())
        y_max = max(xs.max(), ys.max(), zs.max())
        y_span = y_max - y_min
        marker_offset = y_span * 0.06

        if len(peak_times) > 0:
            ax.scatter(peak_times,
                       np.full_like(peak_times, y_max + marker_offset),
                       marker='v', color=peak_color, s=60, zorder=5,
                       label=f'Peak ({len(peak_times)})')

        if len(valley_times) > 0:
            ax.scatter(valley_times,
                       np.full_like(valley_times, y_min - marker_offset),
                       marker='^', color=valley_color, s=60, zorder=5,
                       label=f'Valley ({len(valley_times)})')

        ax.set_ylim(y_min - y_span * 0.15, y_max + y_span * 0.15)

        dur = times[-1] - times[0] if len(times) > 1 else 1
        fs  = len(times) / dur
        primary_tag = '  \u2605 primary' if dev['label'] == primary_label else ''
        ax.set_title(
            f"{dev['label']}{primary_tag}   "
            f"({len(times):,} samples, ~{fs:.0f} Hz)",
            fontsize=11, loc='left')
        ax.set_ylabel("Acceleration\n(native units)", fontsize=9)
        ax.legend(loc='upper right', fontsize=9, ncol=5)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.25)
        ax.set_xlabel("Time (seconds from sync point)", fontsize=9)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    # ------------------------------------------------------------------
    # Stacked bar chart: per-device weighted contribution to combined score
    # ------------------------------------------------------------------
    # The stacked bars show each device's weighted normalised score.
    # Because normalised scores can be negative (below the cross-rep mean),
    # we shift everything up by the minimum so bars always start at zero and
    # the stacking remains visually interpretable. The threshold line is
    # shifted by the same offset.
    n_devices = len(devices)
    n_reps    = len(combined_scores)
    rep_nums  = np.arange(1, n_reps + 1)

    w = np.array(weights, dtype=float)
    w = w / w.sum()

    # Recompute normalised scores for plotting (same logic as score_reps)
    normed = np.zeros_like(per_device_scores)
    for d_i in range(n_devices):
        mu    = float(np.mean(per_device_scores[d_i]))
        sigma = float(np.std(per_device_scores[d_i]))
        sigma = max(sigma, 1e-6)
        normed[d_i] = (per_device_scores[d_i] - mu) / sigma

    weighted = normed * w[:, np.newaxis]   # (n_devices, n_reps)

    # Shift so the minimum combined value sits at y=0
    shift     = combined_scores.min()
    threshold_shifted = threshold - shift

    bottoms = np.zeros(n_reps)
    for d_i, dev in enumerate(devices):
        bar_vals = weighted[d_i] - shift / n_devices   # distribute shift evenly
        color    = device_bar_colors[d_i % len(device_bar_colors)]
        score_ax.bar(rep_nums, bar_vals, bottom=bottoms,
                     color=color, edgecolor='white', linewidth=0.4,
                     alpha=0.85, zorder=3,
                     label=f"{dev['label']} (w={w[d_i]:.2f})")
        bottoms += bar_vals

    # Threshold line and anomaly highlights
    score_ax.axhline(threshold_shifted, color='#c0392b', linewidth=1.5,
                     linestyle='--', zorder=5,
                     label=f'Anomaly threshold ({threshold:.3f})')

    for rep_i in range(n_reps):
        if anomalous[rep_i]:
            score_ax.axvspan(rep_i + 0.5, rep_i + 1.5,
                             color='#e74c3c', alpha=0.08, zorder=0)

    score_ax.set_xlabel("Rep number", fontsize=11)
    score_ax.set_ylabel("Weighted normalised\nDTW score", fontsize=9)
    score_ax.set_title(
        f"Per-rep DTW anomaly score (stacked by device)  —  "
        f"{int(anomalous.sum())} of {n_reps} reps flagged",
        fontsize=11, loc='left')
    score_ax.set_xticks(rep_nums)
    score_ax.legend(loc='upper left', fontsize=9, ncol=n_devices + 1)
    score_ax.grid(True, which='major', linestyle='--', alpha=0.5, axis='y')
    score_ax.set_xlim(0.5, n_reps + 0.5)

    # Colour x-tick labels red for anomalous reps
    for tick, a in zip(score_ax.get_xticklabels(), anomalous):
        tick.set_color('#c0392b' if a else '#333333')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot synchronised accelerometer data from multiple '
                    'devices with rep segmentation and DTW-based anomaly '
                    'detection.')
    parser.add_argument('files', nargs='+', metavar='CSV',
                        help='Accelerometer CSV files (at least 1)')
    parser.add_argument('--labels', nargs='+', metavar='LABEL',
                        default=None,
                        help='Display labels for each device '
                             '(default: filenames)')
    parser.add_argument('--lowpass-hz', type=float, default=5.0,
                        help='Low-pass filter cutoff for peak detection '
                             '(default: 5 Hz)')
    parser.add_argument('--min-separation', type=float, default=None,
                        help='Minimum seconds between peaks/valleys. When '
                             'omitted (default), the value is derived '
                             'automatically from the autocorrelation of the '
                             'filtered magnitude signal, making the script '
                             'exercise-agnostic. Supply this flag to override '
                             'the ACF estimate for edge cases.')
    parser.add_argument('--prominence', type=float, default=0.5,
                        help='Peak prominence threshold as a multiple of '
                             'signal IQR (default: 0.5)')
    parser.add_argument('--primary', type=int, default=0,
                        help='Index of the device to use for rep segmentation '
                             '(default: 0, i.e. first file)')
    parser.add_argument('--template-reps', type=int, default=5,
                        help='Number of early reps to use as good-form '
                             'template (default: 5)')
    parser.add_argument('--anomaly-threshold', type=float, default=2.0,
                        help='Anomaly threshold in standard deviations above '
                             'template-rep mean DTW score (default: 2.0)')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        metavar='W',
                        help='Per-device weights for the combined DTW score '
                             '(one value per file, e.g. --weights 0.5 0.3 0.2). '
                             'Values are normalised to sum to 1 internally. '
                             'Default: equal weights across all devices.')
    parser.add_argument('--save-png', metavar='PATH',
                        help='Save plot as PNG instead of displaying it')
    args = parser.parse_args()

    labels = args.labels or [Path(f).stem for f in args.files]
    if len(labels) != len(args.files):
        parser.error('--labels must have the same number of entries as files')

    if args.primary >= len(args.files):
        parser.error(f'--primary {args.primary} out of range '
                     f'(only {len(args.files)} files provided)')

    # Resolve weights: default to equal; validate length if provided
    if args.weights is None:
        weights = [1.0] * len(args.files)
    else:
        if len(args.weights) != len(args.files):
            parser.error(f'--weights must have the same number of entries as '
                         f'files ({len(args.files)}), got {len(args.weights)}')
        if any(w < 0 for w in args.weights):
            parser.error('--weights values must be non-negative')
        weights = args.weights
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()   # normalise once here for display

    # ------------------------------------------------------------------
    # Load all devices
    # ------------------------------------------------------------------
    all_data = []
    for filepath, label in zip(args.files, labels):
        print(f"\nLoading [{label}]: {filepath}")
        ts, xs, ys, zs = load_device_csv(filepath)
        print(f"    {len(ts):,} samples  |  "
              f"{ts[0]:.3f} → {ts[-1]:.3f}  ({ts[-1]-ts[0]:.1f} s)")
        all_data.append({'label': label,
                         'ts': ts, 'xs': xs, 'ys': ys, 'zs': zs})

    # ------------------------------------------------------------------
    # Find overlapping window: latest start → earliest end
    # ------------------------------------------------------------------
    t_start = max(d['ts'][0]  for d in all_data) + 2
    t_end   = min(d['ts'][-1] for d in all_data) - 2

    if t_start >= t_end:
        raise ValueError(
            f"No overlapping time window found.\n"
            f"  Latest start:  {t_start:.3f}\n"
            f"  Earliest end:  {t_end:.3f}"
        )

    dt_start = datetime.datetime.fromtimestamp(t_start, tz=datetime.timezone.utc)
    print(f"\nSync window: {t_end - t_start:.2f} s  "
          f"(from {dt_start.strftime('%Y-%m-%d %H:%M:%S UTC')})")

    # ------------------------------------------------------------------
    # Trim, normalise time, detect peaks/valleys for every device
    # ------------------------------------------------------------------
    devices = []
    for d in all_data:
        ts, xs, ys, zs = trim_to_window(
            d['ts'], d['xs'], d['ys'], d['zs'], t_start, t_end)
        t0    = ts[0]
        times = ts - t0

        peak_times, valley_times, valley_idx, mag_f, min_sep_used = \
            detect_peaks_valleys(
                times, xs, ys, zs,
                lowpass_hz=args.lowpass_hz,
                min_separation_s=args.min_separation,
                prominence_factor=args.prominence,
            )

        dur = times[-1] if len(times) > 1 else 0
        fs  = len(times) / dur if dur > 0 else 0
        print(f"  [{d['label']}] {len(times):,} samples (~{fs:.0f} Hz)  |  "
              f"min_sep={min_sep_used:.2f}s  |  "
              f"{len(peak_times)} peaks, {len(valley_times)} valleys")

        devices.append({'label':        d['label'],
                        'times':        times,
                        'xs':           xs,
                        'ys':           ys,
                        'zs':           zs,
                        'peak_times':   peak_times,
                        'valley_times': valley_times,
                        'valley_idx':   valley_idx,
                        'mag_f':        mag_f})

    # ------------------------------------------------------------------
    # Rep segmentation on primary device
    # ------------------------------------------------------------------
    primary   = devices[args.primary]
    p_label   = primary['label']
    p_valley  = primary['valley_idx']
    p_mag_f   = primary['mag_f']
    p_times   = primary['times']

    print(f"\nSegmenting reps on primary device: [{p_label}]")
    segments = segment_reps(p_valley,
                             n_samples=len(p_mag_f),
                             n_template=args.template_reps)

    if len(segments) < args.template_reps + 1:
        raise ValueError(
            f"Only {len(segments)} rep(s) detected on [{p_label}]. "
            f"Need at least {args.template_reps + 1} "
            f"({args.template_reps} template + 1 to score).")

    print(f"    {len(segments)} reps detected")
    for i, (s, e) in enumerate(segments):
        dur = p_times[min(e, len(p_times)-1)] - p_times[s]
        print(f"      Rep {i+1:2d}: t={p_times[s]:.2f}s – "
              f"{p_times[min(e, len(p_times)-1)]:.2f}s  ({dur:.2f}s)")

    # ------------------------------------------------------------------
    # Build one template per device, then score all reps across devices
    # ------------------------------------------------------------------
    # Re-order devices so primary is first (plot_all expects this)
    ordered_devices = (
        [devices[args.primary]] +
        [d for i, d in enumerate(devices) if i != args.primary]
    )
    # Weights must follow the same reordering
    ordered_weights = np.concatenate([
        weights[args.primary:args.primary + 1],
        np.delete(weights, args.primary)
    ])

    print(f"\nBuilding good-form templates from first "
          f"{args.template_reps} reps (one per device) ...")
    templates = []
    for dev in ordered_devices:
        print(f"  [{dev['label']}]")
        tmpl = build_template(segments, dev['mag_f'],
                               n_template=args.template_reps)
        templates.append(tmpl)

    print("\nScoring all reps across all devices ...")
    print(f"  Weights: " +
          ", ".join(f"{d['label']}={w:.3f}"
                    for d, w in zip(ordered_devices, ordered_weights)))

    per_device_scores, combined_scores, threshold, anomalous = score_reps(
        segments=segments,
        devices=ordered_devices,
        templates=templates,
        weights=ordered_weights,
        n_template=args.template_reps,
        threshold_sigma=args.anomaly_threshold,
    )

    print(f"  Anomaly threshold (combined): {threshold:.4f}")
    header = "  Rep  |  Combined  |  " + "  ".join(
        f"{d['label']:>10}" for d in ordered_devices)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(len(combined_scores)):
        device_cols = "  ".join(
            f"{per_device_scores[d_i, i]:>10.4f}"
            for d_i in range(len(ordered_devices)))
        flag = "  <- ANOMALOUS" if anomalous[i] else ""
        print(f"  {i+1:3d}  |  {combined_scores[i]:8.4f}  |  "
              f"{device_cols}{flag}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_all(ordered_devices, segments,
             per_device_scores, combined_scores,
             threshold, anomalous,
             weights=ordered_weights,
             primary_label=p_label,
             save_path=args.save_png)


if __name__ == '__main__':
    main()
