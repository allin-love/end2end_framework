import abc

import torch.nn as nn
from torch.fft import ifft2
from util.helper import *
import os
from util import cubicspline
import matplotlib

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
    def __init__(self, image_size, sensor_diameter, lens_diameter, camera_pixel_pitch, d1, d2,
                 window_cropsize=728, require_grad = True, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.sensor_diameter = sensor_diameter
        self.lens_diameter = lens_diameter
        self.camera_pixel_pitch = camera_pixel_pitch
        self.d1 = d1   #物距
        self.d2 = d2   #像距
        self.require_grad = require_grad
        self.delta_z = [0]
        self.wave_resolution = int(np.round(sensor_diameter / self.camera_pixel_pitch))
        self.valid_resolution = int(np.round(self.lens_diameter / self.camera_pixel_pitch))
        self.focal_length = d1 * d2 / (d1 + d2)
        wavelengths = torch.tensor([532e-9]).double()
        self.delta_z = [0]        #点光源相对于聚焦深度平面的z轴偏移
        self.obj_offsets_y = [0]  #点光源在物面相对于视场中心的y轴偏移
        self.obj_offsets_x = [0]  #点光源在物面相对于视场中心的x轴偏移
        self.phase_upsample_factor = 2
        self.window_cropsize = window_cropsize

        self.register_buffer('wavelengths', wavelengths)

        self.param_setup()

    def param_setup(self):
        modulated_phase = torch.zeros((self.valid_resolution//2 // self.phase_upsample_factor)).double()
        self.optim_param = torch.nn.Parameter(modulated_phase, requires_grad=self.require_grad)

    def find_index(self, a, v):
        a = a.squeeze(1).cpu().numpy()
        v = v.cpu().numpy()
        index = np.stack([np.searchsorted(a[i, :], v[i], side='left') - 1 for i in range(a.shape[0])], axis=0)
        return torch.from_numpy(index)

    def phase_bias(self):
        device = self.optim_param.device
        modulated_phase = F.interpolate(self.optim_param.reshape(1, 1, -1),
                      scale_factor=self.phase_upsample_factor, mode='nearest').reshape(-1)
        modulated_phase_log = torch.cat([modulated_phase, torch.zeros((self.valid_resolution // 2), device=device)], dim=0)
        modulated_phase_log = modulated_phase_log.reshape(1, 1, -1)
        r_grid = torch.arange(0, self.valid_resolution, dtype=torch.double, device=device)
        y_coord = torch.arange(0, self.valid_resolution // 2, dtype=torch.double, device=device).reshape(-1, 1) + 0.5
        x_coord = torch.arange(0, self.valid_resolution // 2, dtype=torch.double, device=device).reshape(1, -1) + 0.5
        r_coord = torch.sqrt(y_coord ** 2 + x_coord ** 2).unsqueeze(0)
        r_grid = r_grid.reshape(1, -1)
        ind = self.find_index(r_grid, r_coord).to(device)
        modulated_phase_11 = cubicspline.interp(r_grid, modulated_phase_log, r_coord, ind).float()
        modulated_phase_2d = copy_quadruple(modulated_phase_11)
        return modulated_phase_2d


    def gen_psf(self):
        wave_number = 2 * torch.pi / self.wavelengths

        Dx = (self.camera_pixel_pitch * (torch.arange(self.wave_resolution, device=self.wavelengths.device) - self.wave_resolution // 2)).double()
        DY, DX = torch.meshgrid(Dx, Dx)
        DY = DY
        DX = DX
        mask_shape = 'rect'
        if mask_shape == 'circ':
            aperture_mask = ((DX ** 2 + DY ** 2) < (self.lens_diameter / 2.0) ** 2)
        elif mask_shape == 'rect':
            padding = (self.wave_resolution - self.valid_resolution) // 2
            aperture_mask = torch.ones((self.valid_resolution, self.valid_resolution), device=self.wavelengths.device)
            aperture_mask = F.pad(aperture_mask, (padding, padding, padding, padding))

        lens_phase = - (wave_number * (DX ** 2 + DY ** 2) / (2 * self.focal_length))
        optim_phase_bias = self.phase_bias()
        padding = (self.wave_resolution - self.valid_resolution) // 2
        optim_phase_bias = F.pad(optim_phase_bias, (padding, padding, padding, padding)).squeeze(0).squeeze(0)

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
                    psf_center_x = int(self.wave_resolution // 2 + np.round(self.d2 / self.d1 * obj_offset_x / self.camera_pixel_pitch))
                    psf_center_y = int(self.wave_resolution // 2 + np.round(self.d2 / self.d1 * obj_offset_y / self.camera_pixel_pitch))

                    input_phase = wave_number / 2 / (self.d1 + delta_z) * (
                                torch.square(DX + obj_offset_x) + torch.square(DY + obj_offset_y))
                    field_phase = input_phase + lens_phase + optim_phase_bias
                    # field_phase = wave_number * (torch.sqrt(DX ** 2 + DY ** 2 + (d1+delta_z) ** 2)) - wave_number * (DX ** 2 + DY ** 2) / (2 * focal_length)
                    field_phase %= 2 * torch.pi  # 2pi wrapper

                    field_phase *= aperture_mask
                    field_amp = torch.ones((1, 1), device=self.wavelengths.device) * aperture_mask

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

                    # plt.imshow((psf_region/psf_region.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]).cpu().detach().numpy(), cmap='gray')
                    # plt.title('offset_x={}mm and offset_y={}mm'.format(obj_offset_x * 1e3, obj_offset_y * 1e3))
                    # plt.show()
                    #
                    # plt.imshow(((lens_phase % (2 * torch.pi)) * aperture_mask).cpu().detach().numpy())
                    # plt.title('init phase modulation')
                    # plt.show()
                    #
                    # plt.imshow((((lens_phase + optim_phase_bias)% (2*torch.pi))*aperture_mask).cpu().detach().numpy())
                    # plt.title('optim phase modulation')
                    # plt.show()
                    #
                    # plt.imshow(((optim_phase_bias % (2 * torch.pi)) * aperture_mask).cpu().detach().numpy())
                    # plt.title('optim phase bias')
                    # plt.show()

                    psfs.append(psf_region)

        return torch.stack(psfs,dim=0).unsqueeze(0)

if __name__ == '__main__':
    camera_pixel_pitch = 0.5e-6
    camera = BaseCamera(image_size=384, sensor_diameter=1.8e-3, lens_diameter=0.544e-3, camera_pixel_pitch=0.5e-6,
                        d1=3.35e-3, d2=1.65e-3, num_polynomials=100, window_cropsize=768, require_grad=False)
    # camera = BaseCamera(image_size=384, sensor_diameter=1.8e-3, lens_diameter=0.58e-3, camera_pixel_pitch=0.5e-6,
    #                     d1=3.5e-3, d2=1.7e-3, num_polynomials=100, window_cropsize=768)
    # ckpt_path = "/data4T/hzw/project/FlatScope/Learned_flatscope/20241124_v1/checkpoints/last.ckpt"
    # version = ckpt_path.split('/')[-3]
    # optim_param = torch.load(ckpt_path, map_location={'cuda:0': 'cuda:3'})['state_dict']['camera.optim_param']

    save_psf = False
    if save_psf:
        device = torch.device('cpu')
        camera.obj_offsets_y = [-1.2e-3,-1.0e-3, -0.5e-3, 0, 0.5e-3, 1.0e-3,1.2e-3]
        camera.obj_offsets_x = [-1.2e-3,-1.0e-3, -0.5e-3, 0, 0.5e-3, 1.0e-3,1.2e-3]
    else:
        device = torch.device('cuda:1')
        camera.obj_offsets_y = [-1.2e-3]
        camera.obj_offsets_x = [0]

    camera = camera.to(device)
    # camera.optim_param = torch.nn.Parameter(optim_param.to(device))

    camera.delta_z = [150e-6]
    psfs = camera.gen_psf()

    if save_psf:
        psf_over_fov = np.zeros((len(camera.obj_offsets_y) * 768,len(camera.obj_offsets_x) * 768,3))
        cmapper = matplotlib.cm.get_cmap('magma')
        for i in range(psfs.shape[1]):
            y_ind = int(np.floor(i / len(camera.obj_offsets_x)))
            x_ind = i % len(camera.obj_offsets_x)
            value = (psfs[0,i] / psfs[0,i].max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]).cpu().detach().numpy()
            value = cmapper(value, bytes=True)
            img = value[:, :, :3]
            psf_over_fov[y_ind * 768:(y_ind + 1) * 768, x_ind * 768:(x_ind + 1) * 768] = img
            # psf_over_fov[y_ind*768:(y_ind+1)*768, x_ind*768:(x_ind+1)*768] = (psfs[0,i] / psfs[0,i].max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]).cpu().detach().numpy()
        # im = Image.fromarray((psf_over_fov*255).astype(np.uint8))
        im = Image.fromarray(psf_over_fov.astype(np.uint8))
        dir = '../PSFoverFOV/M={}&L={}mm_SOMMcode'.format(np.around(camera.d2 / camera.d1, 3), np.around(camera.lens_diameter * 1e3, 3))
        if not os.path.exists(dir):
            os.mkdir(dir)
        im.save(os.path.join(dir,'magma_{}_PSF_over_FOV_7×7_deltaz={}.jpg'.format(version,camera.delta_z[0])))

