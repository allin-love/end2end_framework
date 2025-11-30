"""
同心圆二元相位板相机模型 (Binary Phase Plate with Concentric Rings)
更适合光刻加工，相位值仅为 0 或 π

参数化方式：
- 优化环带半径参数 (ring_radii)，而非 Zernike 系数
- 相位板由多个同心环组成，相邻环相位交替为 0 和 π
"""

import abc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2
import os
import matplotlib
import sys

# 动态添加路径以支持 util 模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from util.helper import *
from PIL import Image

try:
    from torch import irfft
    from torch import rfft
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
    """
    同心圆二元相位板相机模型
    
    参数:
        num_rings: 环带数量（可优化参数数量）
        其他参数与 BaseCamera 一致
    """
    def __init__(self, image_size, sensor_diameter, lens_diameter, camera_pixel_pitch, d1, d2, num_rings=20,
                 window_cropsize=768, require_grad=True, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.sensor_diameter = sensor_diameter
        self.lens_diameter = lens_diameter
        self.camera_pixel_pitch = camera_pixel_pitch
        self.d1 = d1  # 物距
        self.d2 = d2  # 像距
        self.require_grad = require_grad
        self.num_rings = num_rings
        
        self.mask_pixel_pitch = 2e-6
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

        self.register_buffer('wavelengths', wavelengths)
        
        # 生成径向坐标网格 (归一化到 [0, 1])
        self._create_radial_grid()
        
        self.param_setup()

    def _create_radial_grid(self):
        """生成归一化径向坐标 (0 表示中心, 1 表示透镜边缘)"""
        dx = torch.arange(self.valid_resolution, dtype=torch.float32) - self.valid_resolution // 2
        Y, X = torch.meshgrid(dx, dx, indexing='ij')
        R = torch.sqrt(X**2 + Y**2)
        R_normalized = R / (self.valid_resolution / 2.0)  # 归一化到 [0, 1]
        R_normalized = torch.clamp(R_normalized, 0, 1)
        self.register_buffer('radial_grid', R_normalized)

    def param_setup(self):
        """
        初始化环带半径参数
        
        策略：初始化为均匀分布的环带 (Fresnel Zone Plate 风格)
        优化器会自动调整这些半径以获得最佳相位分布
        """
        # 初始化: 均匀分布的环带半径 (归一化值, 在 [0, 1] 范围)
        initial_radii = torch.linspace(0.1, 1.0, self.num_rings)
        
        # 为了保证单调性，使用 softmax 后的累积和
        # 实际优化的是 delta_r (相邻环带半径差)
        delta_r = torch.diff(initial_radii, prepend=torch.tensor([0.0]))
        delta_r_logits = torch.log(delta_r + 1e-8)  # 转换为 logits
        
        self.optim_param = torch.nn.Parameter(delta_r_logits, requires_grad=self.require_grad)

    def get_ring_radii(self):
        """
        从可优化参数恢复环带半径 (保证单调递增)
        
        Returns:
            ring_radii: [num_rings], 归一化半径值在 [0, 1]
        """
        delta_r = F.softmax(self.optim_param, dim=0)  # 归一化确保和为1
        ring_radii = torch.cumsum(delta_r, dim=0)
        return ring_radii

    def phase_bias(self):
        """
        生成同心圆二元相位板 (0 或 π)
        
        Returns:
            phase: [1, 1, valid_resolution, valid_resolution], 相位值为 0 或 π
        """
        ring_radii = self.get_ring_radii()
        
        # 初始化相位为 0
        phase = torch.zeros_like(self.radial_grid)
        
        # 根据环带半径分配相位 (交替 0 和 π)
        for i, radius in enumerate(ring_radii):
            mask = self.radial_grid <= radius
            phase[mask] = (i % 2) * np.pi  # 偶数环为 0, 奇数环为 π
        
        # 添加 batch 和 channel 维度
        return phase.unsqueeze(0).unsqueeze(0)

    def gen_psf(self):
        """生成点扩散函数 (PSF)，与 BaseCamera 保持一致的接口"""
        
        wave_number = 2 * torch.pi / self.wavelengths

        Dx = (self.camera_pixel_pitch * (torch.arange(self.wave_resolution, device=self.wavelengths.device) - self.wave_resolution // 2)).float()
        DY, DX = torch.meshgrid(Dx, Dx, indexing='ij')
        
        mask_shape = 'circ'
        if mask_shape == 'circ':
            aperture_mask = ((DX ** 2 + DY ** 2) < (self.lens_diameter / 2.0) ** 2)
        elif mask_shape == 'rect':
            padding = (self.wave_resolution - self.valid_resolution) // 2
            aperture_mask = torch.ones((self.valid_resolution, self.valid_resolution), device=self.wavelengths.device)
            aperture_mask = F.pad(aperture_mask, (padding, padding, padding, padding))

        lens_phase = - (wave_number * (DX ** 2 + DY ** 2) / (2 * self.focal_length))  # 二次透镜相位

        # 获取二元相位板 (0 或 π)
        optim_phase_bias = self.phase_bias().squeeze(0).squeeze(0)
        padding = (self.wave_resolution - self.valid_resolution) // 2
        optim_phase_bias = F.pad(optim_phase_bias, (padding, padding, padding, padding))

        fx = (torch.arange(1, self.wave_resolution + 1, device=self.wavelengths.device) - self.wave_resolution / 2) / self.sensor_diameter
        FY, FX = torch.meshgrid(fx, fx, indexing='ij')
        w1 = torch.square(1 / self.wavelengths) - torch.square(FX) - torch.square(FY)
        w1[w1 < 0] = 0
        w1 = torch.sqrt(w1)
        H_phase = (2 * torch.pi * w1 * self.d2)

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
                    field_amp = torch.ones((1, 1), device=self.wavelengths.device) * aperture_mask

                    field_real = field_amp * torch.cos(field_phase)
                    field_imag = field_amp * torch.sin(field_phase)

                    psf_real, psf_imag = ifft2_apart(
                        ((fftshift(fft2(torch.complex(field_real, field_imag)), dim=[-1, -2])) *
                         torch.complex(torch.cos(H_phase), torch.sin(H_phase))))

                    psf = psf_real ** 2 + psf_imag ** 2

                    psf_region = psf[psf_center_y - self.window_cropsize // 2 + 1:psf_center_y + self.window_cropsize // 2 + 1,
                              psf_center_x - self.window_cropsize // 2 + 1:psf_center_x + self.window_cropsize // 2 + 1]
                    psf_region = psf_region / torch.sum(psf_region, dim=[-2, -1], keepdim=True)

                    psfs.append(psf_region)

        return torch.stack(psfs, dim=0).unsqueeze(0)


if __name__ == '__main__':
    # 测试同心圆二元相位板模型
    camera_pixel_pitch = 1.0e-6
    window_cropsize = 128
    
    camera = BinaryRingsCamera(
        image_size=384, 
        sensor_diameter=5.5e-3, 
        lens_diameter=3.45e-3, 
        camera_pixel_pitch=camera_pixel_pitch,
        d1=65e-3, 
        d2=13.59e-3, 
        num_rings=20,  # 20个环带
        window_cropsize=window_cropsize, 
        require_grad=True
    )
    
    device = torch.device('cpu')
    camera = camera.to(device)
    
    # 可视化初始相位分布
    phase = camera.phase_bias().squeeze().detach().numpy()
    ring_radii = camera.get_ring_radii().detach().numpy()
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(phase, cmap='twilight', vmin=0, vmax=2*np.pi)
    plt.title(f'Binary Phase Plate\n({camera.num_rings} rings)')
    plt.colorbar(label='Phase (rad)')
    
    plt.subplot(1, 3, 2)
    center = phase.shape[0] // 2
    plt.plot(phase[center, :])
    plt.axhline(y=np.pi, color='r', linestyle='--', label='π')
    plt.axhline(y=0, color='b', linestyle='--', label='0')
    plt.xlabel('Pixel Position')
    plt.ylabel('Phase (rad)')
    plt.title('Phase Cross-section')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(ring_radii, 'o-')
    plt.xlabel('Ring Index')
    plt.ylabel('Normalized Radius')
    plt.title('Ring Radii Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('binary_rings_phase_test.png', dpi=150)
    plt.show()
    
    print(f"相位板尺寸: {phase.shape}")
    print(f"环带数量: {camera.num_rings}")
    print(f"环带半径 (归一化): {ring_radii}")
    print(f"可优化参数数量: {camera.optim_param.numel()}")
    
    # 生成 PSF
    print("\n正在生成 PSF...")
    camera.delta_z = [0]  # 只测试焦平面
    psfs = camera.gen_psf()
    print(f"PSF shape: {psfs.shape}")
    
    # 可视化 PSF
    plt.figure(figsize=(6, 6))
    psf_img = psfs[0, 0].detach().numpy()
    plt.imshow(psf_img, cmap='gray')
    plt.title('PSF at Focus')
    plt.colorbar()
    plt.savefig('binary_rings_psf_test.png', dpi=150)
    plt.show()
