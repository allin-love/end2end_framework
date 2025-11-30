import torch

from util import complex
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    from torch.fft import rfft
    from torch.fft import irfft
    def rfft_2(x, d):
        t = rfft2(x)
        return torch.stack((t.real, t.imag), -1)
    def irfft_2(x, d, signal_sizes):
        return irfft2(torch.complex(x[...,0], x[...,1]), s = signal_sizes, dim = (-d, -d+1))
    def rfft_1(x, d):
        t = rfft(x, dim = (-d))
        return torch.stack((t.real, t.imag), -1)
    def irfft_1(x, d, signal_sizes):
        return irfft(torch.complex(x[...,0], x[...,1]), dim = (-d), n=signal_sizes)

def autocorrelation1d_symmetric(h):
    """Compute autocorrelation of a symmetric signal along the last dimension"""
    # Fhsq = complex.abs2(torch.rfft(h, 1))
    Fhsq = complex.abs2(rfft_1(h, 1))

    # a = torch.irfft(torch.stack([Fhsq, torch.zeros_like(Fhsq)], dim=-1), 1, signal_sizes=(h.shape[-1],))
    a = irfft_1(torch.stack([Fhsq, torch.zeros_like(Fhsq)], dim=-1), 1, signal_sizes=h.shape[-1])

    return a / a.max()

def compute_weighting_for_tapering(h):
    """Compute autocorrelation of a symmetric signal along the last two dimension"""
    h_proj0 = h.sum(dim=-2, keepdims=False)
    autocorr_h_proj0 = autocorrelation1d_symmetric(h_proj0).unsqueeze(-2)
    h_proj1 = h.sum(dim=-1, keepdims=False)
    autocorr_h_proj1 = autocorrelation1d_symmetric(h_proj1).unsqueeze(-1)
    return (1 - autocorr_h_proj0) * (1 - autocorr_h_proj1)

def edgetaper3d(img, psf):
    """
    Edge-taper an image with a depth-dependent PSF

    Args:
        img: (B x C x H x W)
        psf: 3d rotationally-symmetric psf (B x C x D x H x W) (i.e. continuous at boundaries)

    Returns:
        Edge-tapered 3D image
    """
    assert (img.dim() == 4)
    assert (psf.dim() == 5)
    psf = psf.mean(dim=-3)
    alpha = compute_weighting_for_tapering(psf)
    blurred_img = irfft_2(
        complex.multiply(rfft_2(img, 2), rfft_2(psf, 2)), 2, signal_sizes=img.shape[-2:]
    )
    return alpha * img + (1 - alpha) * blurred_img