"""Quality gate metrics for the latency-optimization campaign.

PSNR, SSIM, log-spectral distance, and mel-cepstral distortion are pure
torch/torchaudio (both already pipeline dependencies) -- no new installs.
Only LPIPS needs the `lpips` package (a real perceptual metric, no cheap
substitute) -- imported lazily so this module still loads without it.

Every optimized run is compared against a saved baseline at the same seed.
Establish the noise floor (baseline vs itself, two runs, same seed) before
trusting any threshold -- GPU nondeterminism means these are never exactly 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    """a, b: [T, H, W, C] float in [0, max_val]. Higher is better."""
    mse = F.mse_loss(a.float(), b.float()).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val**2 / mse)).item()


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    window = g.t() @ g
    return window.unsqueeze(0).unsqueeze(0)


def ssim(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    """a, b: [T, H, W, C] float in [0, max_val]. Mean SSIM over frames+channels.
    Standard Wang et al. 2004 formulation, 11x11 Gaussian window."""
    device = a.device
    T, H, W, C = a.shape
    a = a.float().permute(0, 3, 1, 2).reshape(T * C, 1, H, W)
    b = b.float().permute(0, 3, 1, 2).reshape(T * C, 1, H, W)
    window = _gaussian_window(device=device)

    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    mu_a = F.conv2d(a, window, padding=5)
    mu_b = F.conv2d(b, window, padding=5)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b

    sigma_a2 = F.conv2d(a * a, window, padding=5) - mu_a2
    sigma_b2 = F.conv2d(b * b, window, padding=5) - mu_b2
    sigma_ab = F.conv2d(a * b, window, padding=5) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))
    return ssim_map.mean().item()


_lpips_model = None


def lpips_distance(a: torch.Tensor, b: torch.Tensor, device: str = "cuda") -> float | None:
    """a, b: [T, H, W, C] float in [0, 1]. Returns None if `lpips` isn't
    installed rather than raising -- caller decides whether that's fatal.
    Lower is better (0 = identical)."""
    global _lpips_model
    try:
        import lpips as lpips_pkg
    except ImportError:
        return None
    if _lpips_model is None:
        _lpips_model = lpips_pkg.LPIPS(net="alex").to(device).eval()
    a = (a.float().permute(0, 3, 1, 2) * 2 - 1).to(device)
    b = (b.float().permute(0, 3, 1, 2) * 2 - 1).to(device)
    with torch.no_grad():
        d = _lpips_model(a, b)
    return d.mean().item()


def frame_to_frame_diff_stats(video: torch.Tensor) -> dict:
    """video: [T, H, W, C]. Flicker/judder check -- compare these stats
    between baseline and optimized, not just per-frame averages (which can
    hide new temporal artifacts)."""
    diffs = (video[1:].float() - video[:-1].float()).abs().mean(dim=(1, 2, 3))
    return {
        "mean_frame_diff": diffs.mean().item(),
        "std_frame_diff": diffs.std().item(),
        "max_frame_diff": diffs.max().item(),
    }


def log_spectral_distance(a: torch.Tensor, b: torch.Tensor, n_fft: int = 1024, hop: int = 256) -> float:
    """a, b: 1D waveform tensors, same sample rate. Lower is better."""
    window = torch.hann_window(n_fft, device=a.device)
    Sa = torch.stft(a.float(), n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs()
    Sb = torch.stft(b.float(), n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs()
    eps = 1e-8
    log_diff = (torch.log10(Sa + eps) - torch.log10(Sb + eps)) ** 2
    return log_diff.mean(dim=0).sqrt().mean().item() * 10.0  # dB-scaled


def mel_cepstral_distortion(a: torch.Tensor, b: torch.Tensor, sample_rate: int, n_mfcc: int = 13) -> float:
    """a, b: 1D waveform tensors, same sample rate. Lower is better.
    Uses torchaudio.transforms.MFCC (already a pipeline dependency).

    n_mels/n_fft passed explicitly: MFCC's own defaults (n_mels=128 against
    a 400-point FFT) leave high mel bins with an all-zero filterbank at
    typical speech sample rates -- torchaudio warns about it, and it would
    silently flatten part of the MCD signal. 40 mels is the standard MFCC
    choice and stays well-conditioned against a 1024-point FFT."""
    import torchaudio

    mfcc_fn = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, n_mfcc=n_mfcc,
        melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40},
    ).to(a.device)
    mfcc_a = mfcc_fn(a.float().unsqueeze(0))[0]  # [n_mfcc, T]
    mfcc_b = mfcc_fn(b.float().unsqueeze(0))[0]
    T = min(mfcc_a.shape[-1], mfcc_b.shape[-1])
    diff = mfcc_a[:, :T] - mfcc_b[:, :T]
    # Standard MCD formula: (10*sqrt(2)/ln(10)) * mean(sqrt(sum(diff^2)))
    return (10.0 * (2**0.5) / torch.log(torch.tensor(10.0))).item() * (diff**2).sum(dim=0).sqrt().mean().item()


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Scale-invariant SDR in dB. Higher is better. Both 1D, same length."""
    estimate = estimate.float() - estimate.float().mean()
    reference = reference.float() - reference.float().mean()
    T = min(estimate.shape[-1], reference.shape[-1])
    estimate, reference = estimate[:T], reference[:T]
    eps = 1e-8
    alpha = (estimate * reference).sum() / (reference**2).sum().clamp_min(eps)
    target = alpha * reference
    noise = estimate - target
    return (10 * torch.log10((target**2).sum().clamp_min(eps) / (noise**2).sum().clamp_min(eps))).item()
