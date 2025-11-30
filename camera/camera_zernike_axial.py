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


class BaseCamera(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, image_size, sensor_diameter, lens_diameter, camera_pixel_pitch, d1, d2, num_polynomials,
                 window_cropsize=768, require_grad = True, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.sensor_diameter = sensor_diameter
        self.lens_diameter = lens_diameter
        self.camera_pixel_pitch = camera_pixel_pitch
        self.d1 = d1  #物距
        self.d2 = d2  #像距
        self.require_grad = require_grad
        self.num_polynomials = num_polynomials
        self.mask_pixel_pitch = 2e-6
        self.mask_size = int(np.round(self.lens_diameter / self.mask_pixel_pitch))
        self.wave_resolution = int(np.round(sensor_diameter / self.camera_pixel_pitch))
        self.valid_resolution = int(np.round(self.lens_diameter / self.camera_pixel_pitch))
        
        # 保证波前调制器分辨率与有效透镜孔径分辨率同奇偶性
        if self.wave_resolution % 2 != self.valid_resolution % 2:
            self.valid_resolution += 1

        self.focal_length = d1 * d2 / (d1 + d2)
        wavelengths = torch.tensor([532e-9]).float()
        self.delta_z = [-0.5e-3,0,0.5e-3]         #点光源相对于聚焦深度平面的z轴偏移
        self.obj_offsets_y = [0]   #点光源在物面相对于视场中心的y轴偏移
        self.obj_offsets_x = [0]   #点光源在物面相对于视场中心的x轴偏移

        self.window_cropsize = window_cropsize

        self.register_buffer('wavelengths', wavelengths)

        zernike_basis = generate_zernike_basis(width=self.mask_size, num_polynomials=self.num_polynomials)
        self.register_buffer('zernike_basis', zernike_basis)
        self.param_setup()

    def param_setup(self):
        ####setup1
        slm_coefficient = torch.zeros(1, self.num_polynomials)

        self.optim_param = torch.nn.Parameter(slm_coefficient, requires_grad=self.require_grad)


    def phase_bias(self):
        optim_phase_bias = generate_phase(self.optim_param, self.zernike_basis)
        return F.interpolate(optim_phase_bias, size=(self.valid_resolution, self.valid_resolution), mode='nearest')

    def gen_psf(self):

        wave_number = 2 * torch.pi / self.wavelengths

        Dx = (self.camera_pixel_pitch * (torch.arange(self.wave_resolution, device=self.wavelengths.device) - self.wave_resolution // 2)).float()
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

        lens_phase = - (wave_number * (DX ** 2 + DY ** 2) / (2 * self.focal_length))  #二次透镜相位
        # lens_phase = - wave_number * (torch.sqrt(DX ** 2 + DY ** 2 + self.focal_length ** 2) - self.focal_length)  #双曲透镜相位

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

        for delta_z in self.delta_z:
            for obj_offset_y in self.obj_offsets_y:
                for obj_offset_x in self.obj_offsets_x:
                    psf_center_x = int(self.wave_resolution // 2 + np.round(self.d2 / (self.d1+delta_z) * obj_offset_x / self.camera_pixel_pitch))
                    psf_center_y = int(self.wave_resolution // 2 + np.round(self.d2 / (self.d1+delta_z) * obj_offset_y / self.camera_pixel_pitch))
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

                    # plt.imshow(psf_region.cpu().detach().numpy(), cmap='gray')
                    # plt.show()


        return torch.stack(psfs,dim=0).unsqueeze(0)

if __name__ == '__main__':
    camera_pixel_pitch = 0.5e-6
    # window_cropsize = 962
    window_cropsize = 128
    # camera = BaseCamera(image_size=384, sensor_diameter=1.4e-3, lens_diameter=0.55e-3, camera_pixel_pitch=0.5e-6,
    #                     d1=3.4e-3, d2=1.6e-3, num_polynomials=100, window_cropsize=window_cropsize, require_grad=False)
    # camera = BaseCamera(image_size=384, sensor_diameter=1.4e-3, lens_diameter=0.545e-3, camera_pixel_pitch=0.5e-6,
    #                     d1=3.4e-3, d2=1.6e-3, num_polynomials=100, window_cropsize=768)
    camera = BaseCamera(image_size=384, sensor_diameter=5.5e-3, lens_diameter=3.45e-3, camera_pixel_pitch=1.0e-6,
                        d1=65e-3, d2=13.59e-3, num_polynomials=100, window_cropsize=window_cropsize, require_grad=False)   
    ckpt_path = r"training_logs/Learned_flatscope/version_2/checkpoints/epoch=48-validation/image_loss=0.0000741.ckpt"
    version = ckpt_path.split('/')[-3]
    optim_param = torch.load(ckpt_path, map_location={'cuda:0': 'cuda:0'})['state_dict']['camera.optim_param']
    # version = 'default'  #to test default PSF

    save_psf = True
    if save_psf:
        device = torch.device('cpu')
        # camera.obj_offsets_y = [-1.0e-3, -0.9e-3,-0.5e-3,0,0.5e-3,0.9e-3,1.0e-3]
        # camera.obj_offsets_x = [-1.0e-3, -0.9e-3,-0.5e-3,0,0.5e-3,0.9e-3,1.0e-3]
        # camera.obj_offsets_y = [-0.95e-3, -0.5e-3, 0, 0.5e-3, 0.95e-3]
        # camera.obj_offsets_x = [-0.95e-3, -0.5e-3, 0, 0.5e-3, 0.95e-3]
        camera.obj_offsets_y = [-3.95e-3, -1.5e-3, 0, 1.5e-3, 3.95e-3]
        camera.obj_offsets_x = [-3.95e-3, -1.5e-3, 0, 1.5e-3, 3.95e-3]
    else:
        device = torch.device('cuda:0')
        camera.obj_offsets_y = [-0.95e-3, 0, 0.95e-3]
        camera.obj_offsets_x = [-0.95e-3, 0, 0.95e-3]

    camera = camera.to(device)


    # camera.optim_param = torch.nn.Parameter(optim_param.to(device))
    camera.delta_z = [-0.5e-3, 0, 0.5e-3]
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
                os.makedirs(dir, exist_ok=True)  # 改为 makedirs,添加 exist_ok=True
            im.save(os.path.join(dir,'magma_{}_PSF_over_FOV_{}×{}_deltaz={}.jpg'.format(version,len(camera.obj_offsets_x),len(camera.obj_offsets_y),delta_z)))
        sys.exit()

