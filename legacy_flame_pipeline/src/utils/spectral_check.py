"""
2D Fast Fourier Transform (FFT) Power-Spectrum & Anisotropy Quality Metric.

Mathematically distinguishes real, directional anatomical facial detail
(wrinkles aligned with Langer's lines and skin tension vectors) from
undifferentiated, isotropic noise or mode-collapsed cobblestone artifacts.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


def compute_2d_power_spectrum(
    field: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes the 2D power spectral density (PSD) of a 2D scalar field
    (e.g., displacement heightfield or normal component).

    Parameters
    ----------
    field : (H, W) float array
    mask : Optional (H, W) boolean or binary float mask

    Returns
    -------
    psd : (H, W) float array of shifted power spectral density (DC at center)
    """
    f = field.astype(np.float64)
    if mask is not None:
        m = (mask > 0.5).astype(np.float64)
        if np.sum(m) > 0:
            mean_val = np.sum(f * m) / np.sum(m)
            f = (f - mean_val) * m
        else:
            f = f - np.mean(f)
    else:
        f = f - np.mean(f)

    # Apply 2D Hann window to attenuate boundary spectral leakage
    h, w = f.shape
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window_2d = np.outer(win_y, win_x)
    f_windowed = f * window_2d

    # 2D FFT with DC shifted to center
    fft2 = np.fft.fft2(f_windowed)
    fft_shifted = np.fft.fftshift(fft2)
    psd = np.abs(fft_shifted) ** 2
    return psd


def compute_angular_anisotropy(
    psd: np.ndarray,
    num_angle_bins: int = 36,
    r_min_ratio: float = 0.05,
    r_max_ratio: float = 0.45,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Measures the angular distribution of power in the frequency domain.

    Real facial wrinkles are directional (anisotropic) because they follow
    skin tension lines, producing energy concentrated along perpendicular angles.
    Isotropic white noise or cobblestone bumps have uniform power across all angles.

    Parameters
    ----------
    psd : (H, W) 2D power spectral density with DC at center
    num_angle_bins : Number of angular sectors in [0, 180) degrees
    r_min_ratio, r_max_ratio : Frequency band bounds as fraction of Nyquist

    Returns
    -------
    anisotropy_ratio : float >= 1.0 (max power / min power across angles)
    angles_deg : (num_angle_bins,) angle centers in degrees
    angular_energy : (num_angle_bins,) integrated power per angle bin
    """
    h, w = psd.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dy = y - cy
    dx = x - cx
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx) % np.pi  # Project to [0, pi)

    max_r = min(cy, cx)
    r_min = r_min_ratio * max_r
    r_max = r_max_ratio * max_r

    band_mask = (r >= r_min) & (r <= r_max)

    bin_edges = np.linspace(0, np.pi, num_angle_bins + 1)
    angular_energy = np.zeros(num_angle_bins, dtype=np.float64)

    for i in range(num_angle_bins):
        bin_m = band_mask & (theta >= bin_edges[i]) & (theta < bin_edges[i + 1])
        if np.any(bin_m):
            angular_energy[i] = np.mean(psd[bin_m])

    # Avoid division by zero
    min_energy = np.min(angular_energy)
    max_energy = np.max(angular_energy)

    if min_energy <= 1e-12:
        anisotropy_ratio = 1.0 if max_energy <= 1e-12 else 10.0
    else:
        anisotropy_ratio = float(max_energy / min_energy)

    angles_deg = np.degrees(0.5 * (bin_edges[:-1] + bin_edges[1:]))
    return anisotropy_ratio, angles_deg, angular_energy


def compute_spectral_slope(
    psd: np.ndarray,
    r_min_ratio: float = 0.05,
    r_max_ratio: float = 0.40,
    num_radial_bins: int = 50,
) -> Tuple[float, float]:
    """
    Computes the radial power-law falloff P(f) ~ 1 / f^alpha.
    Natural surfaces and biological structures exhibit 1.5 <= alpha <= 3.0.
    White/uncorrelated noise has alpha ~ 0.0.

    Returns
    -------
    alpha : Power-law exponent
    r_squared : Goodness of fit
    """
    h, w = psd.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    max_r = min(cy, cx)
    r_min = r_min_ratio * max_r
    r_max = r_max_ratio * max_r

    r_bins = np.linspace(r_min, r_max, num_radial_bins + 1)
    radii = 0.5 * (r_bins[:-1] + r_bins[1:])
    radial_profile = np.zeros(num_radial_bins, dtype=np.float64)

    for i in range(num_radial_bins):
        ring = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if np.any(ring):
            radial_profile[i] = np.mean(psd[ring])

    valid = (radial_profile > 1e-12) & (radii > 0)
    if np.sum(valid) < 5:
        return 0.0, 0.0

    log_r = np.log(radii[valid])
    log_p = np.log(radial_profile[valid])

    # Fit linear slope: log(P) = -alpha * log(r) + c
    poly = np.polyfit(log_r, log_p, 1)
    alpha = float(-poly[0])

    # Compute R^2
    p_pred = poly[0] * log_r + poly[1]
    ss_res = np.sum((log_p - p_pred) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2) + 1e-10
    r_squared = float(1.0 - (ss_res / ss_tot))

    return alpha, r_squared


def verify_displacement_spectral_quality(
    displacement_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
    min_anisotropy_threshold: float = 1.25,
    min_alpha_threshold: float = 0.8,
) -> Dict[str, Union[bool, float, str]]:
    """
    Evaluates whether a displacement map contains structured, directional
    anatomical skin creases or flat isotropic noise.

    Returns dict with metrics and pass/fail verdict.
    """
    disp = displacement_map.squeeze()
    if disp.ndim != 2:
        return {
            'is_valid_anatomical_detail': False,
            'anisotropy_ratio': 1.0,
            'spectral_slope': 0.0,
            'message': f"Expected 2D field, got shape {disp.shape}",
        }

    psd = compute_2d_power_spectrum(disp, mask=mask)
    anisotropy, _, _ = compute_angular_anisotropy(psd)
    alpha, r2 = compute_spectral_slope(psd)

    is_anisotropic = anisotropy >= min_anisotropy_threshold
    has_natural_falloff = alpha >= min_alpha_threshold

    is_valid = is_anisotropic and has_natural_falloff

    if is_valid:
        msg = f"Passed spectral check (Anisotropy: {anisotropy:.2f} >= {min_anisotropy_threshold}, Slope: {alpha:.2f} >= {min_alpha_threshold})"
    elif not is_anisotropic and not has_natural_falloff:
        msg = f"Failed: Isotropic flat noise (Anisotropy: {anisotropy:.2f} < {min_anisotropy_threshold}, Slope: {alpha:.2f} < {min_alpha_threshold})"
    elif not is_anisotropic:
        msg = f"Failed: Undirected/isotropic noise (Anisotropy: {anisotropy:.2f} < {min_anisotropy_threshold})"
    else:
        msg = f"Failed: Unnatural frequency distribution (Slope: {alpha:.2f} < {min_alpha_threshold})"

    return {
        'is_valid_anatomical_detail': is_valid,
        'anisotropy_ratio': float(anisotropy),
        'spectral_slope': float(alpha),
        'r_squared': float(r2),
        'is_anisotropic': is_anisotropic,
        'has_natural_falloff': has_natural_falloff,
        'message': msg,
    }
