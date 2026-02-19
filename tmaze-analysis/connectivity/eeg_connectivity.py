"""
EEG-specific connectivity measures for T-maze analysis.

Includes phase-based connectivity metrics that are more appropriate
for electrophysiological data than amplitude correlations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft


@dataclass
class EEGConnectivityResult:
    """Container for EEG connectivity results."""
    matrix: np.ndarray  # (n_channels, n_channels) or (n_freqs, n_channels, n_channels)
    method: str
    freqs: Optional[np.ndarray] = None
    channel_names: Optional[List[str]] = None
    sfreq: float = 1.0
    metadata: Dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        if self.matrix.ndim == 2:
            return self.matrix.shape[0]
        return self.matrix.shape[1]

    @property
    def is_frequency_resolved(self) -> bool:
        return self.matrix.ndim == 3

    def get_band_connectivity(
        self,
        band: Tuple[float, float]
    ) -> np.ndarray:
        """Get connectivity averaged over frequency band."""
        if not self.is_frequency_resolved or self.freqs is None:
            raise ValueError("Not frequency-resolved")

        mask = (self.freqs >= band[0]) & (self.freqs <= band[1])
        return np.mean(self.matrix[mask], axis=0)


def phase_lag_index(
    data: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 40.0,
    n_fft: Optional[int] = None,
    channel_names: Optional[List[str]] = None
) -> EEGConnectivityResult:
    """
    Compute Phase Lag Index (PLI).

    PLI measures the asymmetry of the distribution of phase differences,
    reducing volume conduction artifacts.

    Parameters
    ----------
    data : np.ndarray
        EEG data (n_channels, n_times) or (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency range
    n_fft : int, optional
        FFT length
    channel_names : List[str], optional
        Channel names

    Returns
    -------
    EEGConnectivityResult
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]  # Add epoch dimension

    n_epochs, n_channels, n_times = data.shape

    if n_fft is None:
        n_fft = n_times

    # Compute instantaneous phase via Hilbert transform
    # First filter in frequency band
    nyq = sfreq / 2
    low = fmin / nyq
    high = min(fmax / nyq, 0.99)

    b, a = signal.butter(4, [low, high], btype='band')

    pli_matrix = np.zeros((n_channels, n_channels))

    for epoch in range(n_epochs):
        # Filter data
        filtered = signal.filtfilt(b, a, data[epoch], axis=1)

        # Compute instantaneous phase
        analytic = signal.hilbert(filtered, axis=1)
        phase = np.angle(analytic)

        # PLI for each pair
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = phase[i] - phase[j]
                # PLI = |mean(sign(sin(phase_diff)))|
                pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                pli_matrix[i, j] += pli
                pli_matrix[j, i] += pli

    # Average across epochs
    pli_matrix /= n_epochs

    return EEGConnectivityResult(
        matrix=pli_matrix,
        method='pli',
        channel_names=channel_names,
        sfreq=sfreq,
        metadata={'fmin': fmin, 'fmax': fmax, 'n_epochs': n_epochs}
    )


def weighted_phase_lag_index(
    data: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 40.0,
    channel_names: Optional[List[str]] = None
) -> EEGConnectivityResult:
    """
    Compute Weighted Phase Lag Index (wPLI).

    wPLI down-weights phase differences near 0 or pi, reducing noise.

    Parameters
    ----------
    data : np.ndarray
        EEG data (n_channels, n_times) or (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency range
    channel_names : List[str], optional
        Channel names

    Returns
    -------
    EEGConnectivityResult
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    n_epochs, n_channels, n_times = data.shape

    # Bandpass filter
    nyq = sfreq / 2
    b, a = signal.butter(4, [fmin/nyq, min(fmax/nyq, 0.99)], btype='band')

    wpli_matrix = np.zeros((n_channels, n_channels))

    for epoch in range(n_epochs):
        filtered = signal.filtfilt(b, a, data[epoch], axis=1)
        analytic = signal.hilbert(filtered, axis=1)
        phase = np.angle(analytic)

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = phase[i] - phase[j]

                # wPLI formula
                imag_csd = np.sin(phase_diff)
                num = np.abs(np.mean(np.abs(imag_csd) * np.sign(imag_csd)))
                denom = np.mean(np.abs(imag_csd))

                wpli = num / denom if denom > 0 else 0
                wpli_matrix[i, j] += wpli
                wpli_matrix[j, i] += wpli

    wpli_matrix /= n_epochs

    return EEGConnectivityResult(
        matrix=wpli_matrix,
        method='wpli',
        channel_names=channel_names,
        sfreq=sfreq,
        metadata={'fmin': fmin, 'fmax': fmax}
    )


def coherence(
    data: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 40.0,
    n_fft: int = 256,
    channel_names: Optional[List[str]] = None
) -> EEGConnectivityResult:
    """
    Compute magnitude-squared coherence.

    Parameters
    ----------
    data : np.ndarray
        EEG data (n_channels, n_times) or (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency range for output
    n_fft : int
        FFT length
    channel_names : List[str], optional
        Channel names

    Returns
    -------
    EEGConnectivityResult
        Frequency-resolved coherence matrix
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    n_epochs, n_channels, n_times = data.shape

    # Compute cross-spectral density
    freqs, csd = _compute_csd(data, sfreq, n_fft)

    # Select frequency range
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    csd = csd[:, :, freq_mask]

    n_freqs = len(freqs)

    # Coherence = |Sxy|^2 / (Sxx * Syy)
    coh = np.zeros((n_freqs, n_channels, n_channels))

    for f in range(n_freqs):
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    coh[f, i, j] = 1.0
                else:
                    num = np.abs(csd[i, j, f]) ** 2
                    denom = np.real(csd[i, i, f]) * np.real(csd[j, j, f])
                    coh[f, i, j] = num / denom if denom > 0 else 0

    return EEGConnectivityResult(
        matrix=coh,
        method='coherence',
        freqs=freqs,
        channel_names=channel_names,
        sfreq=sfreq,
        metadata={'n_fft': n_fft}
    )


def imaginary_coherence(
    data: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 40.0,
    n_fft: int = 256,
    channel_names: Optional[List[str]] = None
) -> EEGConnectivityResult:
    """
    Compute imaginary part of coherence.

    Imaginary coherence is insensitive to volume conduction
    (zero-lag correlations).

    Parameters
    ----------
    data : np.ndarray
        EEG data
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency range
    n_fft : int
        FFT length
    channel_names : List[str], optional
        Channel names

    Returns
    -------
    EEGConnectivityResult
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    n_epochs, n_channels, n_times = data.shape

    freqs, csd = _compute_csd(data, sfreq, n_fft)

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    csd = csd[:, :, freq_mask]

    n_freqs = len(freqs)
    icoh = np.zeros((n_freqs, n_channels, n_channels))

    for f in range(n_freqs):
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    icoh[f, i, j] = 0
                else:
                    # Imaginary coherence
                    num = np.imag(csd[i, j, f])
                    denom = np.sqrt(np.real(csd[i, i, f]) * np.real(csd[j, j, f]))
                    icoh[f, i, j] = np.abs(num / denom) if denom > 0 else 0

    return EEGConnectivityResult(
        matrix=icoh,
        method='imaginary_coherence',
        freqs=freqs,
        channel_names=channel_names,
        sfreq=sfreq,
        metadata={'n_fft': n_fft}
    )


def phase_synchrony(
    data: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 40.0,
    channel_names: Optional[List[str]] = None
) -> EEGConnectivityResult:
    """
    Compute Phase Locking Value (PLV).

    PLV measures consistency of phase difference across time/trials.

    Parameters
    ----------
    data : np.ndarray
        EEG data (n_epochs, n_channels, n_times) for trial-based PLV
        or (n_channels, n_times) for single-trial instantaneous PLV
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency range
    channel_names : List[str], optional
        Channel names

    Returns
    -------
    EEGConnectivityResult
    """
    if data.ndim == 2:
        # Single trial: return time-resolved PLV (averaged over time)
        data = data[np.newaxis, :, :]

    n_epochs, n_channels, n_times = data.shape

    # Bandpass filter
    nyq = sfreq / 2
    b, a = signal.butter(4, [fmin/nyq, min(fmax/nyq, 0.99)], btype='band')

    plv_matrix = np.zeros((n_channels, n_channels))

    # Get phases for all epochs
    all_phases = np.zeros((n_epochs, n_channels, n_times))

    for epoch in range(n_epochs):
        filtered = signal.filtfilt(b, a, data[epoch], axis=1)
        analytic = signal.hilbert(filtered, axis=1)
        all_phases[epoch] = np.angle(analytic)

    # PLV across trials (at each time point, then average)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = all_phases[:, i, :] - all_phases[:, j, :]

            # PLV = |mean(exp(i * phase_diff))|
            # Average across epochs, then time
            plv_time = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))
            plv = np.mean(plv_time)

            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv

    np.fill_diagonal(plv_matrix, 1.0)

    return EEGConnectivityResult(
        matrix=plv_matrix,
        method='plv',
        channel_names=channel_names,
        sfreq=sfreq,
        metadata={'fmin': fmin, 'fmax': fmax, 'n_epochs': n_epochs}
    )


def _compute_csd(
    data: np.ndarray,
    sfreq: float,
    n_fft: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cross-spectral density matrix using Welch's method."""
    n_epochs, n_channels, n_times = data.shape

    # Compute PSD/CSD
    freqs = np.fft.rfftfreq(n_fft, 1/sfreq)
    n_freqs = len(freqs)

    csd = np.zeros((n_channels, n_channels, n_freqs), dtype=complex)

    for epoch in range(n_epochs):
        # FFT of each channel
        fft_data = np.fft.rfft(data[epoch], n=n_fft, axis=1)

        # Cross-spectral density
        for i in range(n_channels):
            for j in range(n_channels):
                csd[i, j] += fft_data[i] * np.conj(fft_data[j])

    csd /= n_epochs

    return freqs, csd


def compute_connectivity_bands(
    data: np.ndarray,
    sfreq: float,
    method: str = 'wpli',
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    channel_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute connectivity in standard frequency bands.

    Parameters
    ----------
    data : np.ndarray
        EEG data
    sfreq : float
        Sampling frequency
    method : str
        'pli', 'wpli', 'coherence', 'plv'
    bands : Dict[str, Tuple[float, float]], optional
        Band definitions {name: (fmin, fmax)}
    channel_names : List[str], optional
        Channel names

    Returns
    -------
    Dict[str, np.ndarray]
        Connectivity matrix for each band
    """
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

    results = {}

    for band_name, (fmin, fmax) in bands.items():
        if method == 'pli':
            conn = phase_lag_index(data, sfreq, fmin, fmax, channel_names=channel_names)
        elif method == 'wpli':
            conn = weighted_phase_lag_index(data, sfreq, fmin, fmax, channel_names=channel_names)
        elif method == 'coherence':
            conn = coherence(data, sfreq, fmin, fmax, channel_names=channel_names)
            # Average across frequencies in band
            conn.matrix = np.mean(conn.matrix, axis=0)
        elif method == 'plv':
            conn = phase_synchrony(data, sfreq, fmin, fmax, channel_names=channel_names)
        else:
            raise ValueError(f"Unknown method: {method}")

        results[band_name] = conn.matrix

    return results


def connectivity_condition_contrast(
    data1: np.ndarray,
    data2: np.ndarray,
    sfreq: float,
    method: str = 'wpli',
    fmin: float = 1.0,
    fmax: float = 40.0,
    n_permutations: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare connectivity between conditions.

    Parameters
    ----------
    data1, data2 : np.ndarray
        EEG data for two conditions
    sfreq : float
        Sampling frequency
    method : str
        Connectivity method
    fmin, fmax : float
        Frequency range
    n_permutations : int
        Number of permutations

    Returns
    -------
    diff : np.ndarray
        Connectivity difference (cond1 - cond2)
    p_values : np.ndarray
        P-values for each edge
    """
    if method == 'wpli':
        conn1 = weighted_phase_lag_index(data1, sfreq, fmin, fmax)
        conn2 = weighted_phase_lag_index(data2, sfreq, fmin, fmax)
    elif method == 'pli':
        conn1 = phase_lag_index(data1, sfreq, fmin, fmax)
        conn2 = phase_lag_index(data2, sfreq, fmin, fmax)
    elif method == 'plv':
        conn1 = phase_synchrony(data1, sfreq, fmin, fmax)
        conn2 = phase_synchrony(data2, sfreq, fmin, fmax)
    else:
        raise ValueError(f"Unknown method: {method}")

    diff = conn1.matrix - conn2.matrix
    n_channels = diff.shape[0]

    # Permutation test
    combined = np.concatenate([data1, data2], axis=0)
    n1 = data1.shape[0]

    null_diffs = np.zeros((n_permutations, n_channels, n_channels))

    for perm in range(n_permutations):
        perm_idx = np.random.permutation(combined.shape[0])
        perm1 = combined[perm_idx[:n1]]
        perm2 = combined[perm_idx[n1:]]

        if method == 'wpli':
            c1 = weighted_phase_lag_index(perm1, sfreq, fmin, fmax)
            c2 = weighted_phase_lag_index(perm2, sfreq, fmin, fmax)
        elif method == 'pli':
            c1 = phase_lag_index(perm1, sfreq, fmin, fmax)
            c2 = phase_lag_index(perm2, sfreq, fmin, fmax)
        else:
            c1 = phase_synchrony(perm1, sfreq, fmin, fmax)
            c2 = phase_synchrony(perm2, sfreq, fmin, fmax)

        null_diffs[perm] = c1.matrix - c2.matrix

    p_values = np.mean(np.abs(null_diffs) >= np.abs(diff), axis=0)
    p_values = np.maximum(p_values, 1 / (n_permutations + 1))

    return diff, p_values
