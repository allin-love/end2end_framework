"""Binary phase-plate camera with concentric ring parameterization."""

import abc
from pathlib import Path
import sys
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.fft import fft2, ifft2

# Ensure project root (for util imports) is discoverable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.helper import *

try:
    from torch import irfft  # type: ignore[attr-defined]
    from torch import rfft  # type: ignore[attr-defined]
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2

    def rfft_2(x, d):
        t = rfft2(x)
        return torch.stack((t.real, t.imag), -1)

    def irfft_2(x, d, signal_sizes):
        return irfft2(torch.complex(x[..., 0], x[..., 1]), s=signal_sizes, dim=(-d, -d + 1))

    def fft2_apart(x):
        t = fft2(x)
        return torch.stack((t.real, t.imag), -1)

    def ifft2_apart(x):
        t = ifft2(x)
        return t.real, t.imag


class BinaryRingsCamera(nn.Module, metaclass=abc.ABCMeta):
    """Concentric-ring binary phase plate with trainable ring radii."""
    def __init__(self, image_size, sensor_diameter, lens_diameter, camera_pixel_pitch, d1, d2, num_rings=20,
                 window_cropsize=768, require_grad=True, mask_pixel_pitch=None, ring_softness=60.0,
                 psf_crop_size=None, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.sensor_diameter = sensor_diameter
        self.lens_diameter = lens_diameter
        self.camera_pixel_pitch = camera_pixel_pitch
        self.d1 = d1  # 物距
        self.d2 = d2  # 像距
        self.require_grad = require_grad
        self.num_rings = num_rings
        self.ring_softness = ring_softness

        # Optical sampling pitch (default 2 µm unless overridden by caller)
        self.mask_pixel_pitch = 2e-6 if mask_pixel_pitch is None else mask_pixel_pitch
        self.mask_size = int(np.round(self.lens_diameter / self.mask_pixel_pitch))
        self.wave_resolution = int(np.round(sensor_diameter / self.camera_pixel_pitch))
        self.valid_resolution = int(np.round(self.lens_diameter / self.camera_pixel_pitch))
        
        # 保证波前调制器分辨率与有效透镜孔径分辨率同奇偶性
        if self.wave_resolution % 2 != self.valid_resolution % 2:
            self.valid_resolution += 1

        self.focal_length = d1 * d2 / (d1 + d2)
        wavelengths = torch.tensor([532e-9]).float()
        self.delta_z = [-0.5e-3, 0, 0.5e-3]         # 点光源相对于聚焦深度平面的z轴偏移
        self.obj_offsets_y = [0]   # 点光源在物面相对于视场中心的y轴偏移
        self.obj_offsets_x = [0]   # 点光源在物面相对于视场中心的x轴偏移

        self.window_cropsize = window_cropsize
        self.psf_crop_size = min(self.window_cropsize, psf_crop_size if psf_crop_size is not None else window_cropsize)

        self.wavelengths: Tensor
        self.register_buffer('wavelengths', wavelengths)
        
        # 生成径向坐标网格 (归一化到 [0, 1])
        self._create_radial_grid()
        
        self.param_setup()
        self._prepare_wave_domain()

    def _create_radial_grid(self):
        """Pre-compute normalized radius map (0=center, 1=edge)."""
        dx = torch.arange(self.valid_resolution, dtype=torch.float32) - self.valid_resolution // 2
        Y, X = torch.meshgrid(dx, dx, indexing='ij')
        R = torch.sqrt(X**2 + Y**2)
        R_normalized = R / (self.valid_resolution / 2.0)  # 归一化到 [0, 1]
        R_normalized = torch.clamp(R_normalized, 0, 1)
        self.radial_grid: Tensor
        self.register_buffer('radial_grid', R_normalized)

    def param_setup(self):
        """Initialize ring spacing logits (monotonic radii after softmax)."""
        initial_radii = torch.linspace(0.1, 1.0, self.num_rings)
        
        # 为了保证单调性，使用 softmax 后的累积和
        # 实际优化的是 delta_r (相邻环带半径差)
        delta_r = torch.diff(initial_radii, prepend=torch.tensor([0.0]))
        delta_r_logits = torch.log(delta_r + 1e-8)  # 转换为 logits
        
        self.optim_param = Parameter(delta_r_logits, requires_grad=self.require_grad)

    def _prepare_wave_domain(self):
        """Pre-compute coordinate grids and optical buffers reused each iteration."""
        device = self.wavelengths.device
        dtype = torch.float32

        coords = (torch.arange(self.wave_resolution, device=device, dtype=dtype) -
                  self.wave_resolution // 2) * self.camera_pixel_pitch
        DY, DX = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('DY_grid', DY)
        self.register_buffer('DX_grid', DX)

        radius_sq = DX ** 2 + DY ** 2
        self.register_buffer('radius_squared', radius_sq)

        aperture_mask = (radius_sq < (self.lens_diameter / 2.0) ** 2).to(dtype)
        self.register_buffer('aperture_mask', aperture_mask)

        padding = (self.wave_resolution - self.valid_resolution) // 2
        self.pad_tuple = (padding, padding, padding, padding)

        wave_number = (2 * torch.pi / self.wavelengths).view(-1, 1, 1)
        self.register_buffer('wave_number', wave_number)

        freq = ((torch.arange(1, self.wave_resolution + 1, device=device, dtype=dtype) -
                 self.wave_resolution / 2) / self.sensor_diameter)
        FY, FX = torch.meshgrid(freq, freq, indexing='ij')
        self.register_buffer('FY_grid', FY)
        self.register_buffer('FX_grid', FX)

        w1 = torch.square(1 / self.wavelengths.view(-1, 1, 1)) - torch.square(FX) - torch.square(FY)
        w1 = torch.sqrt(torch.clamp(w1, min=0.0))
        H_phase = (2 * torch.pi * w1 * self.d2)
        self.register_buffer('H_phase', H_phase)

    def get_ring_radii(self):
        """Recover monotonically increasing radii in [0, 1]."""
        delta_r = F.softmax(self.optim_param, dim=0)  # 归一化确保和为1
        ring_radii = torch.cumsum(delta_r, dim=0)
        return ring_radii

    def _smooth_ring(self, inner_radius: Tensor, outer_radius: Tensor) -> Tensor:
        """Generate a differentiable annulus mask between inner and outer radii."""
        softness = self.radial_grid.new_full((), self.ring_softness)
        inner_edge = torch.sigmoid((self.radial_grid - inner_radius) * softness)
        outer_edge = torch.sigmoid((self.radial_grid - outer_radius) * softness)
        return torch.clamp(outer_edge - inner_edge, min=0.0, max=1.0)

    def phase_bias(self):
        """Return smooth 0/π phase map with alternating rings."""
        device = self.radial_grid.device
        dtype = self.radial_grid.dtype
        ring_radii = self.get_ring_radii().to(device=device, dtype=dtype)

        phase = torch.zeros_like(self.radial_grid)

        prev_radius = torch.tensor(0.0, device=device, dtype=dtype)
        for i, radius in enumerate(ring_radii):
            ring_mask = self._smooth_ring(prev_radius, radius)
            phase = phase + ((i % 2) * np.pi) * ring_mask
            prev_radius = radius

        return phase.unsqueeze(0).unsqueeze(0)

    def gen_psf(self):
        """生成点扩散函数 (PSF)，与 BaseCamera 保持一致的接口"""

        DX = cast(Tensor, self.DX_grid)
        DY = cast(Tensor, self.DY_grid)
        aperture_mask = cast(Tensor, self.aperture_mask)
        wave_number = cast(Tensor, self.wave_number)
        radius_squared = cast(Tensor, self.radius_squared)
        H_phase = cast(Tensor, self.H_phase)
        lens_phase = -(wave_number * radius_squared / (2 * self.focal_length))  # 二次透镜相位

        # 获取二元相位板 (0 或 π)
        optim_phase_bias = self.phase_bias().squeeze(0).squeeze(0)
        optim_phase_bias = F.pad(optim_phase_bias, self.pad_tuple)

        psfs = []

        for delta_z in self.delta_z:
            for obj_offset_y in self.obj_offsets_y:
                for obj_offset_x in self.obj_offsets_x:
                    psf_center_x = int(self.wave_resolution // 2 + np.round(self.d2 / (self.d1 + delta_z) * obj_offset_x / self.camera_pixel_pitch))
                    psf_center_y = int(self.wave_resolution // 2 + np.round(self.d2 / (self.d1 + delta_z) * obj_offset_y / self.camera_pixel_pitch))
                    input_phase = wave_number / 2 / (self.d1 + delta_z) * (
                        torch.square(DX + obj_offset_x) + torch.square(DY + obj_offset_y))
                    field_phase = input_phase + lens_phase + optim_phase_bias
                    field_phase %= 2 * torch.pi  # 2pi wrapper

                    field_phase *= aperture_mask
                    field_amp = aperture_mask

                    field_real = field_amp * torch.cos(field_phase)
                    field_imag = field_amp * torch.sin(field_phase)

                    psf_real, psf_imag = ifft2_apart(
                        ((fftshift(fft2(torch.complex(field_real, field_imag)), dim=[-1, -2])) *
                        torch.complex(torch.cos(H_phase), torch.sin(H_phase))))

                    psf = psf_real ** 2 + psf_imag ** 2

                    y_start = psf_center_y - self.window_cropsize // 2 + 1
                    y_end = psf_center_y + self.window_cropsize // 2 + 1
                    x_start = psf_center_x - self.window_cropsize // 2 + 1
                    x_end = psf_center_x + self.window_cropsize // 2 + 1
                    psf_region = psf[:, y_start:y_end, x_start:x_end]
                    if self.psf_crop_size < psf_region.shape[-1]:
                        crop = self.psf_crop_size
                        cy = psf_region.shape[-2] // 2
                        cx = psf_region.shape[-1] // 2
                        y0 = cy - crop // 2
                        x0 = cx - crop // 2
                        psf_region = psf_region[:, y0:y0 + crop, x0:x0 + crop]
                    psf_region = psf_region / torch.sum(psf_region, dim=[-2, -1], keepdim=True)

                    psfs.append(psf_region)

        psfs = torch.stack(psfs, dim=0)  # (cases, n_lambda, H, W)
        cases, n_lambda, h, w = psfs.shape
        psfs = psfs.view(1, cases * n_lambda, h, w)
        return psfs


def _demo_output_dir() -> Path:
    out_dir = PROJECT_ROOT / "artifacts" / "binary_rings_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _run_demo():
    camera = BinaryRingsCamera(
        image_size=384,
        sensor_diameter=5.5e-3,
        lens_diameter=3.45e-3,
        camera_pixel_pitch=1.0e-6,
        d1=65e-3,
        d2=13.59e-3,
        num_rings=20,
        window_cropsize=128,
        require_grad=True,
    ).cpu()

    phase = camera.phase_bias().squeeze().detach().numpy()
    ring_radii = camera.get_ring_radii().detach().numpy()
    out_dir = _demo_output_dir()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(phase, cmap="twilight", vmin=0, vmax=2 * np.pi)
    axes[0].set_title(f"Binary Phase Plate\n({camera.num_rings} rings)")
    fig.colorbar(axes[0].images[0], ax=axes[0], label="Phase (rad)")

    center = phase.shape[0] // 2
    axes[1].plot(phase[center, :])
    axes[1].axhline(y=np.pi, color="r", linestyle="--", label="π")
    axes[1].axhline(y=0, color="b", linestyle="--", label="0")
    axes[1].set(xlabel="Pixel Position", ylabel="Phase (rad)", title="Phase Cross-section")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(ring_radii, "o-")
    axes[2].set(xlabel="Ring Index", ylabel="Normalized Radius", title="Ring Radii Distribution")
    axes[2].grid(True)

    fig.tight_layout()
    phase_path = out_dir / "phase_demo.png"
    fig.savefig(str(phase_path), dpi=150)
    plt.close(fig)

    camera.delta_z = [0]
    psf_img = camera.gen_psf()[0, 0].detach().numpy()
    fig_psf, ax_psf = plt.subplots(1, 1, figsize=(5, 5))
    ax_psf.imshow(psf_img, cmap="gray")
    ax_psf.set_title("PSF at Focus")
    fig_psf.colorbar(ax_psf.images[0], ax=ax_psf)
    psf_path = out_dir / "psf_demo.png"
    fig_psf.savefig(str(psf_path), dpi=150)
    plt.close(fig_psf)

    print(f"Demo artifacts written to {out_dir}")


if __name__ == "__main__":
    _run_demo()
