"""
Optimized Binary Phase Plate Camera
Refactored for speed, convergence stability, and backward compatibility.
Final robust fix for Inplace Operations and Memory Views.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.fft import fft2, ifft2, fftshift,ifftshift
from typing import cast

# 辅助函数：复数操作
def fft2_apart(real, imag):
    complex_input = torch.complex(real, imag)
    ft = fft2(complex_input)
    return ft.real, ft.imag

def ifft2_apart(real, imag):
    complex_input = torch.complex(real, imag)
    ift = ifft2(complex_input)
    return ift.real, ift.imag

class BinaryRingsCamera(nn.Module):
    def __init__(self, image_size, sensor_diameter, lens_diameter, camera_pixel_pitch, d1, d2, num_rings=100,
                 window_cropsize=384, require_grad=True, train_downsample=4, mask_pixel_pitch=2e-6, **kwargs):
        """
        Args:
            train_downsample (int): 训练时的降采样倍率（建议 4 或 8）。
            mask_pixel_pitch (float): 仅用于兼容日志记录，实际计算使用 native/train pixel pitch。
        """
        super().__init__()
        
        # --- 兼容性属性 ---
        self.mask_pixel_pitch = mask_pixel_pitch 
        
        # --- 基础参数 ---
        self.image_size = image_size
        self.sensor_diameter = sensor_diameter
        self.lens_diameter = lens_diameter
        # 原始高精度像素尺寸
        self.native_pixel_pitch = camera_pixel_pitch  
        # 训练时使用的低精度像素尺寸
        self.train_pixel_pitch = camera_pixel_pitch * train_downsample
        self.train_downsample = train_downsample
        
        self.d1 = d1
        self.d2 = d2
        self.num_rings = num_rings
        self.window_cropsize = window_cropsize
        self.require_grad = require_grad
        
        # 光学参数
        self.focal_length = d1 * d2 / (d1 + d2)
        # 波长 (绿光)
        self.register_buffer('wavelengths', torch.tensor([532e-9]).float())
        
        # 深度平面 (Depth Planes)
        self.delta_z = [-0.5e-3, 0, 0.5e-3]
        self.obj_offsets_y = [0]
        self.obj_offsets_x = [0]
        
        # --- 初始化参数 ---
        self.param_setup()
        
        # --- 预计算坐标网格 (Lazy Loading) ---
        self.grid_cache = {} 

    def param_setup(self):
        """
        使用直接线性参数化。
        """
        # 初始化为近似 Axicon 的线性分布
        init_width = 1.0 / self.num_rings # 0.01
        
        # --- 【修改点】不再取 log，直接存物理宽度 ---
        # 加入一点点随机噪声防止死锁
        random_noise = torch.rand(self.num_rings) * 0.2 + 0.9
        self.optim_param = nn.Parameter(
            torch.tensor(init_width) * random_noise, 
            requires_grad=self.require_grad
        )
        
        # Sharpness 保持 2.0
        self.sharpness = nn.Parameter(torch.tensor(2.0), requires_grad=self.require_grad)

    def get_ring_radii(self):
        """
        计算归一化半径 [0, 1]。
        """
        # --- 【修改点】使用 abs() 保证宽度为正，梯度导数为 1 或 -1 (不衰减) ---
        widths = torch.abs(self.optim_param)
        
        # 累加得到半径
        radii = torch.cumsum(widths, dim=0)
        # 归一化到最大半径为 1.0
        radii = radii / radii[-1]
        return radii
    def _get_grids(self, device, dtype, pixel_pitch):
        """
        生成或获取缓存的坐标网格。
        """
        key = f"{pixel_pitch}_{device}"
        if key in self.grid_cache:
            return self.grid_cache[key]
        
        # 计算分辨率
        res = int(np.ceil(self.sensor_diameter / pixel_pitch))
        if res % 2 != 0: res += 1
        
        # 1. 空间坐标 (x, y)
        coord = (torch.arange(res, device=device, dtype=dtype) - res // 2) * pixel_pitch
        Y, X = torch.meshgrid(coord, coord, indexing='ij')
        R2 = X**2 + Y**2
        R = torch.sqrt(R2)
        
        # 2. 孔径掩膜
        mask = (R2 < (self.lens_diameter / 2.0)**2).float()

        wavelengths = cast(torch.Tensor, self.wavelengths)

        # 3. 透镜相位 (Lens Phase)
        two_pi = torch.full_like(wavelengths, 2 * math.pi)
        k = two_pi / wavelengths
        # 近轴近似二次相位
        lens_phase = - (k * R2) / (2 * self.focal_length)
        
        # 4. 菲涅尔传播核 (H_phase)
        freq = (torch.arange(res, device=device, dtype=dtype) - res // 2) / (res * pixel_pitch)
        FY, FX = torch.meshgrid(freq, freq, indexing='ij')
        rho_sq = FX**2 + FY**2
        inv_lambda_sq = torch.reciprocal(wavelengths) ** 2
        root_term = torch.sqrt(torch.clamp(inv_lambda_sq - rho_sq, min=0))
        H_term = 2 * torch.pi * root_term * self.d2
        
        # 归一化半径图
        norm_R = R / (self.lens_diameter / 2.0)
        
        cache_item = {
            'X': X, 'Y': Y, 'mask': mask, 
            'lens_phase': lens_phase, 'H_term': H_term,
            'norm_R': norm_R, 'res': res
        }
        self.grid_cache[key] = cache_item
        return cache_item

    def gen_phase_plate(self, norm_R):
        """使用 sigmoid 计数实现稳定的二元相位板。"""
        radii = self.get_ring_radii()

        beta = F.softplus(self.sharpness)

        # Sum of sigmoids: count how many rings have been crossed at each radius.
        ring_counts = torch.sigmoid(beta * (norm_R.unsqueeze(-1) - radii))
        ring_counts = ring_counts.sum(dim=-1)

        # Convert counts into phase by checking parity (softly via cosine).
        phase_map = 0.5 * (1 - torch.cos(ring_counts * torch.pi)) * torch.pi

        return phase_map

    # --- 【兼容接口】 ---
    def phase_bias(self):
        """
        兼容性接口：为 TensorBoard 日志提供相位图。
        返回形状: [1, 1, H, W]
        """
        wavelengths = cast(torch.Tensor, self.wavelengths)
        pitch = self.native_pixel_pitch
        grids = self._get_grids(wavelengths.device, wavelengths.dtype, pitch)
        phase = self.gen_phase_plate(grids['norm_R'])
        return phase.unsqueeze(0).unsqueeze(0)

    def forward(self):
        return self.gen_psf()

    def gen_psf(self):
        # 1. 决定分辨率 (训练降采样，验证全分辨)
        if self.training:
            pitch = self.train_pixel_pitch
        else:
            pitch = self.native_pixel_pitch
            
        wavelengths = cast(torch.Tensor, self.wavelengths)
        grids = self._get_grids(wavelengths.device, wavelengths.dtype, pitch)
        
        # 2. 生成相位板
        optim_phase = self.gen_phase_plate(grids['norm_R'])
        
        # 3. 组合瞳孔函数
        total_phase = grids['lens_phase'] + optim_phase
        pupil_complex = grids['mask'] * torch.exp(1j * total_phase)
        
        # 4. 传播准备
        pupil_fft = fftshift(fft2(pupil_complex), dim=(-2, -1))
        
        psfs = []
        two_pi = torch.full_like(wavelengths, 2 * math.pi)
        k = two_pi / wavelengths
        
        for delta_z in self.delta_z:
            # 离焦项
            dist = self.d1 + delta_z
            input_curvature = (k * (grids['X']**2 + grids['Y']**2)) / (2 * dist)
            
            # 更新瞳孔
            curr_pupil = pupil_complex * torch.exp(1j * input_curvature)
            curr_pupil_fft = fftshift(fft2(curr_pupil), dim=(-2, -1))
            sensor_field_fft = curr_pupil_fft * torch.exp(1j * grids['H_term'])
            sensor_field = ifft2(ifftshift(sensor_field_fft, dim=(-2, -1)))
            
            # 强度
            psf = torch.abs(sensor_field)**2
            
            # 归一化
            psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + 1e-8)
            
            # 裁剪中心 (Center Crop)
            h, w = psf.shape[-2], psf.shape[-1]
            if self.training:
                crop = self.window_cropsize // self.train_downsample
            else:
                crop = self.window_cropsize
                
            cy, cx = h // 2, w // 2
            
            # --- 【关键修复点】 ---
            # 使用 .clone() 强制断开与大张量的内存共享关系
            # 这解决了 "modified by inplace operation" 的问题
            psf_crop = psf[..., cy - crop//2 : cy + crop//2, cx - crop//2 : cx + crop//2].clone()
            
            psfs.append(psf_crop)
            
        # 堆叠维度: [Depth, H, W] -> [1, Depth, H, W]
        # 使用 stack (复制数据) + unsqueeze
        return torch.stack(psfs, dim=0).unsqueeze(0)