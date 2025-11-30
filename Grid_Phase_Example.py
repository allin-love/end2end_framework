import abc
import profile

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.fft import ifft2
from util.helper import *
import os
import matplotlib
import scipy.io as scio
import sys
import random
from scipy.signal import fftconvolve

import math
from util.refractive_index import refractive_index_glass_bk7 # 假设你的DOE材料是BK7

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
'''
def cross_coherence_matrix(psfs, order='pos-major', eps=1e-8, reduce='mean_abs'):
    """
    计算每个视场位置的 Cross-coherence 矩阵（深度×深度）。
    psfs: [B, 27, H, W]
    order: 'pos-major' 表示 (pos0_d0,pos0_d1,pos0_d2, pos1_d0,...)；
           'depth-major' 表示 (d0_pos0,...,d0_pos8, d1_pos0,...,d2_pos8)
    reduce: 'mean_abs'  -> 以频域 |C_ij(u,v)| 的平均值作为标量相干度
            'phasecorr' -> 反变换后的相位相关峰值（phase correlation peak）

    返回:
      sim_mat: [B, 9, 3, 3]  每个视场位置的深度间相干度矩阵（标量）
    """
    B, C, H, W = psfs.shape
    assert C == 27, "C 应为 27 = 9*3"
    # 重排为 [B, 9, 3, H, W]
    if order == 'pos-major':
        x = psfs.view(B, 9, 3, H, W)
    elif order == 'depth-major':
        x = psfs.view(B, 3, 9, H, W).permute(0, 2, 1, 3, 4).contiguous()
    else:
        raise ValueError("order 只能是 'pos-major' 或 'depth-major'")

    # 2D FFT
    X = torch.fft.fft2(x, dim=(-2, -1))  # [B, 9, 3, H, W]
    # 归一化到单位幅度（避免幅值影响，仅比较相位一致性）
    mag = torch.abs(X).clamp_min(eps)
    Xn = X / mag  # [B,9,3,H,W], 每个深度的谱相位单位化

    # 构造 3×3 互谱相干: C_ij(u,v) = Xn_i * conj(Xn_j)
    # 先把维度展平，方便批量乘： [B,9,3,HW]
    HW = H * W
    Xn_flat = Xn.reshape(B, 9, 3, HW)
    # 两两相乘得到 [B,9,3,3,HW]
    C = Xn_flat.unsqueeze(3) * torch.conj(Xn_flat.unsqueeze(2))

    if reduce == 'mean_abs':
        # 以 |C_ij| 在频域上的平均作为标量相干度
        sim_mat = torch.mean(torch.abs(C), dim=-1)  # [B,9,3,3]
    elif reduce == 'phasecorr':
        # 相位相关：ifft2(C_ij) 的峰值（对每个 i,j 做一次 ifft2）
        # 先还原到 [B,9,3,3,H,W]
        C_maps = C.view(B, 9, 3, 3, H, W)
        # ifft2
        c_spatial = torch.fft.ifft2(C_maps, dim=(-2, -1))
        # 取幅值最大值作为峰值
        sim_mat = torch.amax(torch.abs(c_spatial), dim=(-2, -1))  # [B,9,3,3]
    else:
        raise ValueError("reduce 只能是 'mean_abs' 或 'phasecorr'")

    return sim_mat  # [B,9,3,3]
'''

def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    with np.errstate(divide='ignore', invalid='ignore'):
        out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out

def ncc_depth_matrix_per_position(psfs, D, P, mode='full', reduce='max'):
    """
    psfs: ndarray, shape [D*P, H, W]  (depth-major)
    D: depth_planes, P: fov_positions
    mode: 'full' | 'same' | 'valid'  (传给 normxcorr2)
    reduce: 'max' -> 取相关图最大值; 'center' -> 取中心值
    return:
      mats: [P, D, D]  每个位置的深度间相似度矩阵(标量)
    """
    assert psfs.ndim == 4, "输入应为 [B,D*P, H, W]"
    B, DP, H, W = psfs.shape
    assert B == 1, "Batch应等于1"
    assert DP == D * P

    # 重排: [D,P,H,W] -> [P,D,H,W]
    psfs_pd = psfs.reshape(D, P, H, W).transpose(1, 0, 2, 3).copy()

    mats = np.zeros((P, D, D), dtype=np.float32)

    for p in range(P):
        for i in range(D):
            for j in range(D):
                ncc_map = normxcorr2(psfs_pd[p, i], psfs_pd[p, j], mode=mode)
                if reduce == 'max':
                    mats[p, i, j] = float(ncc_map.max())
                elif reduce == 'center':
                    # 取中心值（不同 mode 下中心索引不同，这里以 full 为例）
                    cx = ncc_map.shape[0] // 2
                    cy = ncc_map.shape[1] // 2
                    mats[p, i, j] = float(ncc_map[cx, cy])
                else:
                    raise ValueError("reduce must be 'max' or 'center'")
    return mats  # [P, D, D]

def ncc_against_centralPSF(psfs, D, P, mode='full', reduce='max'):
    """
    以 idx0 = (D*P)//2 的 PSF 为模板，计算其与所有 PSF 的 normxcorr2 相似度。
    psfs: [D*P, H, W]（depth-major）
    D: 深度面个数；P: 视场位置个数
    mode: 传给 normxcorr2（'full'/'same'/'valid'）
    reduce: 'max' -> 取相关图最大值；'center' -> 取中心值
    返回:
      scores_flat: [D*P]（按 depth-major 排列）
      scores_mat : [D, P]（重排后便于查看）
      idx0, (d0, p0): 模板索引及其对应的 (depth_idx, pos_idx)
    """
    assert psfs.ndim == 4 and psfs.shape[1] == D*P
    B, DP, H, W = psfs.shape
    psfs = psfs.squeeze(0)
    idx0 = (D*P) // 2
    T = psfs[idx0]

    scores = np.zeros(DP, dtype=np.float32)
    for k in range(DP):
        ncc_map = normxcorr2(T, psfs[k], mode=mode)
        if reduce == 'max':
            scores[k] = float(ncc_map.max())
        elif reduce == 'center':
            cx, cy = ncc_map.shape[0]//2, ncc_map.shape[1]//2
            scores[k] = float(ncc_map[cx, cy])
        else:
            raise ValueError("reduce must be 'max' or 'center'")

    scores_mat = scores.reshape(D, P)  # depth-major -> [D, P]
    d0, p0 = divmod(idx0, P)          # 模板对应的 (depth_idx, pos_idx)
    return scores_mat

def cross_coherence_matrix(psfs, order='pos-major', fov_num=9, center=False, eps=1e-8):
    """
    严格归一化的 cross-coherence（实质为余弦相似度），对角线恒为 1。
    输入:
        psfs: [B, 27, H, W]，27 = 9个视场位置 × 3个深度
        order:
            'pos-major'   : (pos0_d0,pos0_d1,pos0_d2, pos1_d0,...)  默认
            'depth-major' : (d0_pos0,...,d0_pos8, d1_pos0,...,d2_pos8)
        center: 是否先做零均值（去掉DC；有些场景更稳）
        eps: 数值稳定项
    输出:
        C: [B, 9, 3, 3]  每个视场位置的深度×深度相似度矩阵
    """
    B, C, H, W = psfs.shape
    assert C % fov_num == 0, "通道数必须是fov_num的倍数"
    depth_planes = C // fov_num

    # 重排到 [B, 9, 3, H, W]
    if order == 'pos-major':
        x = psfs.view(B, fov_num, depth_planes, H, W)
    elif order == 'depth-major':
        x = psfs.view(B, depth_planes, fov_num, H, W).permute(0, 2, 1, 3, 4).contiguous()
    else:
        raise ValueError("order 必须为 'pos-major' 或 'depth-major'")

    # 展平空间维度 -> [B,9,3,HW]
    x = x.view(B, fov_num, depth_planes, -1)

    # 去均值
    mean = x.mean(dim=-1, keepdim=True)
    x_centered = x - mean

    # L2 范数
    norms = torch.linalg.vector_norm(x_centered, ord=2, dim=-1, keepdim=True).clamp_min(eps)
    x_norm = x_centered / norms  # 单位化

    # 零位移互相关（即余弦相似度）
    C = torch.matmul(x_norm, x_norm.transpose(-1, -2))  # [B,9,3,3]

    return C

class BaseCamera(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, image_size, sensor_diameter, lens_diameter, camera_pixel_pitch, d1, d2, num_polynomials, delta_z = 0,
                 window_cropsize=768, require_grad = True, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.sensor_diameter = sensor_diameter
        self.lens_diameter = lens_diameter
        self.camera_pixel_pitch = camera_pixel_pitch
        self.d1 = d1
        self.d2 = d2
        self.delta_z = [-100e-6,0,250e-6]
        self.require_grad = require_grad
        self.num_polynomials = num_polynomials
        self.mask_pixel_pitch = 2e-6
        self.mask_size = int(np.round(self.lens_diameter / self.mask_pixel_pitch))
        self.wave_resolution = int(np.round(sensor_diameter / self.camera_pixel_pitch))
        self.valid_resolution = int(np.round(self.lens_diameter / self.camera_pixel_pitch))

        if self.wave_resolution % 2 != self.valid_resolution % 2:
            self.valid_resolution += 1

        self.focal_length = d1 * d2 / (d1 + d2)
        wavelengths = torch.tensor([520e-9]).double()
        self.obj_offsets_y = [-0.95e-3, 0, 0.95e-3]
        self.obj_offsets_x = [-0.95e-3, 0, 0.95e-3]
        # self.sample_points = [[-1e-3, 0.5e-3], [-1e-3, 0], [-1e-3, -0.5e-3], [1e-3, 0.5e-3], [1e-3, 0], [1e-3,-0.5e-3],
        #                       [0, 0],[-0.5e-3, -1e-3], [0,-1e-3], [0.5e-3, -1e-3], [-0.5e-3, 1e-3], [0, 1e-3], [0.5e-3,1e-3]]
        self.window_cropsize = window_cropsize

        self.register_buffer('wavelengths', wavelengths)

        # zernike_basis = generate_zernike_basis(width=self.valid_resolution, num_polynomials=self.num_polynomials)
        zernike_basis = generate_zernike_basis(width=self.mask_size, num_polynomials=self.num_polynomials)
        self.register_buffer('zernike_basis', zernike_basis)
        self.param_setup()

    def param_setup(self):
        ####setup1
        slm_coefficient = torch.zeros(1, self.num_polynomials)

        ####setup2
        # ckpt_path = "/data4T/hzw/project/FlatScope/Learned_flatscope/20250922_v1/checkpoints/epoch=29-val_loss=-0.0012399.ckpt"
        # slm_coefficient = torch.load(ckpt_path, map_location={'cuda:0': 'cuda:0'})['state_dict']['camera.optim_param']

        self.optim_param = torch.nn.Parameter(slm_coefficient, requires_grad=self.require_grad)


    def phase_bias(self):
        optim_phase_bias = generate_phase(self.optim_param, self.zernike_basis)
        return F.interpolate(optim_phase_bias, size=(self.valid_resolution, self.valid_resolution), mode='nearest')

    def gen_psf(self):

        wave_number = 2 * torch.pi / self.wavelengths

        Dx = (self.camera_pixel_pitch * (torch.arange(self.wave_resolution, device=self.wavelengths.device) - self.wave_resolution // 2)).double()
        DY, DX = torch.meshgrid(Dx, Dx)
        DY = DY
        DX = DX
        mask_shape = 'circ'
        if mask_shape == 'circ':
            aperture_mask = ((DX ** 2 + DY ** 2) < (self.lens_diameter / 2.0) ** 2)
        elif mask_shape == 'rect':
            padding = (self.wave_resolution - self.valid_resolution) // 2
            aperture_mask = torch.ones((self.valid_resolution, self.valid_resolution), device=self.wavelengths.device)
            aperture_mask = F.pad(aperture_mask, (padding, padding, padding, padding))

        # lens_phase = - (wave_number * (DX ** 2 + DY ** 2) / (2 * self.focal_length))  #二次相位
        lens_phase = - wave_number * (torch.sqrt(DX ** 2 + DY ** 2 + self.focal_length ** 2) - self.focal_length)  #双曲相位

        optim_phase_bias = self.phase_bias().squeeze(0).squeeze(0)
        padding = (self.wave_resolution - self.valid_resolution) // 2
        optim_phase_bias = F.pad(optim_phase_bias, (padding, padding, padding, padding))

        fx = (torch.arange(1, self.wave_resolution + 1,device=self.wavelengths.device) - self.wave_resolution / 2) / self.sensor_diameter
        FY, FX = torch.meshgrid(fx, fx)
        w1 = torch.square(1 / self.wavelengths) - torch.square(FX) - torch.square(FY)
        w1[w1 < 0] = 0
        w1 = torch.sqrt(w1)
        H_phase = (2 * torch.pi * w1 * self.d2)

        psfs = []



        phase_profile = lens_phase + optim_phase_bias.squeeze(0).squeeze(0) #如果是要把这个phase_profile用到ZEMAX中的网格相位中，则不用进行2π wrap
        phase_profile *= aperture_mask
        plt.imshow(phase_profile.cpu().detach().numpy())
        plt.title('optim phase profile')
        plt.colorbar()
        plt.show()
        # ######For ZEMAX # 保存为 txt，把这个二维相位面展平为一维矩阵，每行记录一个位置的相位
        # phase_flat = phase_profile.flatten()
        # with open(r"D:/user_doc/Remote/DOE/data/fabrication/20251112_test1.txt", 'w') as f:
        #     for value in phase_flat:
        #         f.write(f"{value.item():.3f}\n")


        # ###### For ZEMAX (修正版 - 导出为 Grid Sag .dat 文件) ######
        
        # 1. 获取 "相位图" (单位: 弧度)
        # phase_profile_radians = (lens_phase + optim_phase_bias.squeeze(0).squeeze(0)) #
        phase_profile_radians = optim_phase_bias.squeeze(0).squeeze(0)
        phase_profile_radians *= aperture_mask
        phase_profile_radians_numpy = phase_profile_radians.detach().cpu().numpy()

        # 2. 将 "相位 (radians)" 转换为 "矢高 (mm)"
        # Zemax 需要的是物理高度 (Sag)，单位是 mm
        
        wavelength_meters = self.wavelengths[0].item() # 你的仿真波长 (e.g., 520e-9)
        n = refractive_index_glass_bk7(wavelength_meters) # 获取材料折射率
        
        wave_number_in_air = (2 * math.pi) / wavelength_meters
        delta_n = n - 1 # 假设材料到空气

        # 计算高度图 (单位: 米)
        height_map_meters = phase_profile_radians_numpy / (wave_number_in_air * delta_n)
        # 转换为Zemax需要的毫米 (mm)
        height_map_mm = height_map_meters * 1000.0

        # 3. 准备 Zemax 表头参数
        Nx = height_map_mm.shape[1] # 数据点数量 (列数)
        Ny = height_map_mm.shape[0] # 数据点数量 (行数)
        
        # 数据间隔 (dx, dy)，单位 mm
        dx_mm = self.camera_pixel_pitch * 1000.0 # 你的仿真像素间隔
        dy_mm = dx_mm

        unit_code = 0 # 0 = mm
        x_decenter = 0.0 #
        y_decenter = 0.0 #

        # 4. 写入Zemax .dat 文件
        # (使用你原来的文件路径，但后缀改为 .dat)
        output_filename_dat = r"D:/user_doc/Remote/DOE/data/fabrication/20251112_test1.dat"

        print(f"正在导出 Zemax Grid Sag 文件到: {output_filename_dat}")
        print(f"  Grid (Nx, Ny): {Nx}, {Ny}")
        print(f"  Spacing (dx, dy): {dx_mm:.6f} mm, {dy_mm:.6f} mm")
        print(f"  Sag Range (mm): {height_map_mm.min():.6f} to {height_map_mm.max():.6f}")

        with open(output_filename_dat, 'w') as f:
            # 写入7个数字的表头
            f.write(f"{Nx} {Ny} {dx_mm} {dy_mm} {unit_code} {x_decenter} {y_decenter}\n")
            
            # 使用 np.savetxt 高效写入 2D 矩阵
            np.savetxt(f, height_map_mm, fmt='%.8e')
            
        print("Zemax 文件导出完成。")


        for delta_z in self.delta_z:
            for obj_offset_y in self.obj_offsets_y:
                for obj_offset_x in self.obj_offsets_x:
                    psf_center_x = int(self.wave_resolution // 2 + np.round(self.d2 / (self.d1+delta_z) * obj_offset_x / self.camera_pixel_pitch))
                    psf_center_y = int(self.wave_resolution // 2 + np.round(self.d2 / (self.d1+delta_z) * obj_offset_y / self.camera_pixel_pitch))
                    input_phase = wave_number / 2 / (self.d1 + delta_z) * (
                                torch.square(DX + obj_offset_x) + torch.square(DY + obj_offset_y))
                    field_phase = input_phase + lens_phase + optim_phase_bias
                    # field_phase = wave_number * (torch.sqrt(DX ** 2 + DY ** 2 + (d1+delta_z) ** 2)) - wave_number * (DX ** 2 + DY ** 2) / (2 * focal_length)
                    field_phase %= 2 * torch.pi  # 2pi wrapper
                    # if self.require_grad:
                    #     field_phase += fab_noise  #1120添加
                    field_phase *= aperture_mask
                    field_amp = torch.ones((1, 1), device=self.wavelengths.device) * aperture_mask
                    # field_phase = field_phase * circ_mask
                    # plt.imshow(field_phase.cpu().numpy())
                    # plt.show()

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

        return torch.stack(psfs,dim=0).unsqueeze(0)

if __name__ == '__main__':
    camera_pixel_pitch = 0.5e-6
    # window_cropsize = 962
    window_cropsize = 128
    # camera = BaseCamera(image_size=384, sensor_diameter=1.4e-3, lens_diameter=0.55e-3, camera_pixel_pitch=0.5e-6,
    #                     d1=3.4e-3, d2=1.6e-3, num_polynomials=100, window_cropsize=window_cropsize, require_grad=False)
    camera = BaseCamera(image_size=128, sensor_diameter=8.625e-3, lens_diameter=3.45e-3, camera_pixel_pitch=1.12e-6,
                        d1=65e-3, d2=13.59e-3, num_polynomials=100, window_cropsize=window_cropsize, require_grad=False)     
    ckpt_path = r"training_logs_deconv_edof/Learned_flatscope/version_1/checkpoints/last.ckpt"
    version = ckpt_path.split('/')[-3]
    optim_param = torch.load(ckpt_path, map_location={'cuda:0': 'cuda:0'})['state_dict']['camera.optim_param']

    save_psf = False
    if save_psf:
        device = torch.device('cpu')
        camera.obj_offsets_y = [-3.95e-3, -1.5e-3, 0, 1.5e-3, 3.95e-3]
        camera.obj_offsets_x = [-3.95e-3, -1.5e-3, 0, 1.5e-3, 3.95e-3]
    else:
        device = torch.device('cuda:0')
        camera.obj_offsets_y = [-0.95e-3, 0, 0.95e-3]
        camera.obj_offsets_x = [-0.95e-3, 0, 0.95e-3]

    camera = camera.to(device)

    camera.optim_param = torch.nn.Parameter(optim_param.to(device))
    camera.delta_z = [-100e-6]
    psfs = camera.gen_psf()
    if save_psf:
        cmapper = matplotlib.cm.get_cmap('magma')
        for k in range(len(camera.delta_z)):
            delta_z = camera.delta_z[k]
            index_start = k*len(camera.obj_offsets_y)*len(camera.obj_offsets_x)
            index_end = (k+1)*len(camera.obj_offsets_y)*len(camera.obj_offsets_x)
            psf_over_fov = np.zeros((len(camera.obj_offsets_y) * window_cropsize,len(camera.obj_offsets_x) * window_cropsize,3))

            # cmapper = matplotlib.cm.get_cmap('gray')
            for i in range(index_start,index_end):
                pos_index = i % (len(camera.obj_offsets_x)*len(camera.obj_offsets_y))
                y_ind = int(np.floor(pos_index / len(camera.obj_offsets_x)))
                x_ind = pos_index % len(camera.obj_offsets_x)
                value = (psfs[0,i] / psfs[0,i].max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]).cpu().detach().numpy()
                value = cmapper(value, bytes=True)
                img = value[:, :, :3]
                psf_over_fov[y_ind * window_cropsize:(y_ind + 1) * window_cropsize, x_ind * window_cropsize:(x_ind + 1) * window_cropsize] = img
                # psf_over_fov[y_ind*768:(y_ind+1)*768, x_ind*768:(x_ind+1)*768] = (psfs[0,i] / psfs[0,i].max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]).cpu().detach().numpy()
            # im = Image.fromarray((psf_over_fov*255).astype(np.uint8))
            im = Image.fromarray(psf_over_fov.astype(np.uint8))
            dir = '../PSFoverFOV/M={}&L={}mm_SOMMcode'.format(np.around(camera.d2 / camera.d1, 3), np.around(camera.lens_diameter * 1e3, 3))
            if not os.path.exists(dir):
                os.mkdir(dir)
            im.save(os.path.join(dir,'magma_{}_PSF_over_FOV_{}×{}_deltaz={}.jpg'.format(version,len(camera.obj_offsets_x),len(camera.obj_offsets_y),delta_z)))



