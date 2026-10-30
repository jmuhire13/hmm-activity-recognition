"""Feature extraction and normalization for one window of accelerometer and gyroscope readings.

The functions are the ones used in notebooks/feature_extraction.ipynb and
notebooks/normalization.ipynb, collected here so that new recordings are processed in exactly
the same way as the training windows. tests/test_window_features.py checks that this module
reproduces the saved training feature tables.
"""

import numpy as np
import pandas as pd

SENSORS = ["acc", "gyro"]
AXES = ["x", "y", "z"]
LOW_FREQUENCY_CUTOFF_HZ = 3.0
FEATURE_NAMES = ["std", "corr", "jerk_ratio", "dom_freq", "low_freq_share", "spec_centroid"]
FEATURE_COLUMNS = [f"{sensor}_{name}" for sensor in SENSORS for name in FEATURE_NAMES]
LOG10_COLUMNS = ["acc_std", "gyro_std"]
LOG1P_COLUMNS = ["acc_dom_freq", "gyro_dom_freq"]
LOG_FLOOR = 1e-6  # keeps log10 defined if a window is perfectly flat


def magnitude(signal):
    """Combined strength of the three axes at each reading."""
    return np.sqrt((signal ** 2).sum(axis=1))


def total_std(signal):
    """Standard deviation over all three axes: square root of the summed variances."""
    return float(np.sqrt(signal.var(axis=0).sum()))


def mean_abs_correlation(signal):
    """Average absolute correlation over the three axis pairs.

    The correlation of a perfectly flat axis is undefined; it is counted as 0.
    """
    values = []
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        if signal[:, first].std() == 0 or signal[:, second].std() == 0:
            values.append(0.0)
        else:
            values.append(abs(np.corrcoef(signal[:, first], signal[:, second])[0, 1]))
    return float(np.mean(values))


def jerk_ratio(signal):
    """Standard deviation of the reading-to-reading change of the magnitude, divided by
    the standard deviation of the magnitude. Small for smooth signals, large for jagged ones."""
    strength = magnitude(signal)
    spread = strength.std()
    return float(np.diff(strength).std() / spread) if spread > 0 else 0.0


def _power_spectrum(signal, sampling_rate_hz):
    """Frequencies in Hz and FFT power of the mean-removed magnitude, zero-frequency bin excluded."""
    strength = magnitude(signal)
    power = np.abs(np.fft.rfft(strength - strength.mean())) ** 2
    frequencies = np.fft.rfftfreq(len(strength), d=1 / sampling_rate_hz)
    return frequencies[1:], power[1:]


def dominant_frequency(signal, sampling_rate_hz):
    """Frequency in Hz of the strongest component of the magnitude."""
    frequencies, power = _power_spectrum(signal, sampling_rate_hz)
    return float(frequencies[power.argmax()])


def low_frequency_share(signal, sampling_rate_hz, cutoff_hz=LOW_FREQUENCY_CUTOFF_HZ):
    """Fraction of the spectral power that lies below `cutoff_hz` (0 if there is no power)."""
    frequencies, power = _power_spectrum(signal, sampling_rate_hz)
    total = power.sum()
    return float(power[frequencies < cutoff_hz].sum() / total) if total > 0 else 0.0


def spectral_centroid(signal, sampling_rate_hz):
    """Power-weighted average frequency in Hz (0 if there is no power)."""
    frequencies, power = _power_spectrum(signal, sampling_rate_hz)
    total = power.sum()
    return float((frequencies * power).sum() / total) if total > 0 else 0.0


def window_features(window):
    """Return a dictionary with the 12 feature values of one window.

    `window` is a table with a time_s column and the columns acc_x, acc_y, acc_z, gyro_x,
    gyro_y and gyro_z. The sampling rate is taken from the median time step of the window.
    """
    sampling_rate_hz = 1 / window["time_s"].diff().median()
    features = {}
    for sensor in SENSORS:
        signal = window[[f"{sensor}_{axis}" for axis in AXES]].to_numpy()
        features[f"{sensor}_std"] = total_std(signal)
        features[f"{sensor}_corr"] = mean_abs_correlation(signal)
        features[f"{sensor}_jerk_ratio"] = jerk_ratio(signal)
        features[f"{sensor}_dom_freq"] = dominant_frequency(signal, sampling_rate_hz)
        features[f"{sensor}_low_freq_share"] = low_frequency_share(signal, sampling_rate_hz)
        features[f"{sensor}_spec_centroid"] = spectral_centroid(signal, sampling_rate_hz)
    return features


def log_transform(table):
    """Apply log10 to the strength features and log1p to the dominant frequencies."""
    out = table.copy()
    for column in LOG10_COLUMNS:
        out[column] = np.log10(np.maximum(table[column], LOG_FLOOR))
    for column in LOG1P_COLUMNS:
        out[column] = np.log1p(table[column])
    return out


def apply_scaler(table, scaler):
    """Log-transform a feature table and standardize it with a scaler fitted on training data.

    `scaler` is the dictionary saved in data/features/scaler.json. It must never be refitted
    on new recordings.
    """
    transformed = log_transform(table)
    for column in FEATURE_COLUMNS:
        transformed[column] = (transformed[column] - scaler["mean"][column]) / scaler["std"][column]
    return transformed
