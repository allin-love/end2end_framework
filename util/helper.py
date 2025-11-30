import torch
import torch.nn.functional as F
import torch.nn as nn
from aotools.functions import zernikeArray
from torch.fft import fft2, fftshift, ifftshift, irfftn, rfftn, ifft2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchgeometry as tgm

def imresize(img, size):
    return F.interpolate(img, size=size)

def binnning(factor, img):
    temp = torch.zeros((img.shape[0], img.shape[1], img.shape[-2]//factor, img.shape[-1]//factor), device=img.device)
    for y in range(0, factor):
        for x in range(0, factor):
            temp += img[..., y::factor, x::factor]
    return temp

def crop_image(tensor, width):
    start_idx = (tensor.shape[-2] - width) // 2
    end_idx = start_idx + width
    # Extract central crop
    cropped_tensor = tensor[..., start_idx:end_idx, start_idx:end_idx]
    return cropped_tensor

def generate_zernike_basis(width, num_polynomials=100):
    # num_polynomials = 28  ## orders of Zernike
    # width = 256  ## image size
    # device=torch.device('cuda:0')
    # num_polynomials = (zern_order*(zern_order+1))//2
    zernike_diam = np.ceil(width * np.sqrt(2))  # radius of 256
    zernike = zernikeArray(num_polynomials, zernike_diam)
    zernike = torch.FloatTensor(zernike)
    crop_zernike = crop_image(zernike, width)
    return crop_zernike

def generate_phase(zern_alpha, zernike_basis, no_translation=False):
    zern_alpha_no_translation = zern_alpha.clone().to(zern_alpha.device)

    if no_translation:
        zern_alpha_no_translation[:, :3] = 0

    weights = 2 * zern_alpha_no_translation[..., None, None]
    zernike_basis_expand = zernike_basis.unsqueeze(0).clone()
    weighted_zernike = zernike_basis_expand * weights
    phs = weighted_zernike.sum(dim=1, keepdim=True)

    return phs

def complex_matmul(a, b, groups = 1):
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])

def fft_2xPad_Conv2D(signal, kernel, groups=1):
    size = signal.shape[-1]

    signal_fr = rfftn(signal, dim=[-2, -1], s=[2 * size, 2 * size])
    kernel_fr = rfftn(kernel, dim=[-2, -1], s=[2 * size, 2 * size])

    output_fr = complex_matmul(signal_fr, kernel_fr, groups)

    output = irfftn(output_fr, dim=[-2, -1], s=[-1, -1])

    s2 = size//2
    output = output[:, :, s2:-s2, s2:-s2]

    return output

def conv_psf(obj, psf, use_FFT=True, mask=None):


    n_batch = obj.shape[0]
    n_slm, n_mask, h, w = psf.shape

    psf = psf.view(-1, h, w).unsqueeze(1)

    if use_FFT:
        y = fft_2xPad_Conv2D(obj, psf)
    else:
        y = F.conv2d(obj, psf, padding='same')

    y = y.view(n_batch, n_slm, n_mask, h, w)
    if n_slm == 1:
        y = y.squeeze(1)
    if mask is None:
        return y
    else:
        mask = mask.unsqueeze(0)
        return (y * mask).sum(dim=2, keepdim=True)

# def binning(psfs):
    # psfs_downsample = torch.zeros((psfs.shape[0], psfs.shape[1],psfs.shape[-2] // 2, psfs.shape[-1] // 2), device=psfs.device)
    # for i in range(psfs_downsample.shape[-2]):  # 540
    #     for j in range(psfs_downsample.shape[-1]):  # 960
    #         if i % 2 == 0 and j % 2 == 0:
    #             psfs_downsample[..., i,j] = (psfs[..., i * 2, j * 2] + psfs[..., i * 2, j * 2 + 2] + psfs[..., i * 2 + 2, j * 2]
    #                           + psfs[..., i * 2 + 2, j * 2 + 2]) / 4.
    #         elif i % 2 == 0 and j % 2 == 1:
    #             psfs_downsample[..., i, j] = (psfs[..., i * 2, j * 2 - 1] + psfs[..., i * 2, j * 2 + 1] + psfs[..., i * 2 + 1, j * 2 - 1]
    #                                           + psfs[..., i * 2 + 1, j * 2 + 1]) / 4.
    #         elif i % 2 == 1 and j % 2 == 0:
    #             psfs_downsample[..., i, j] = (psfs[..., i * 2 - 1, j * 2] + psfs[..., i * 2 - 1, j * 2 + 2] + psfs[..., i * 2 + 1, j * 2]
    #                                           + psfs[..., i * 2 + 1, j * 2 + 2]) / 4.
    #         elif i % 2 == 1 and j % 2 == 1:
    #             psfs_downsample[..., i, j] = (psfs[..., i * 2 - 1, j * 2 - 1] + psfs[..., i * 2 - 1, j * 2 + 1]
    #                                           + psfs[..., i * 2 + 1, j * 2 - 1] + psfs[..., i * 2 + 1, j * 2 + 1]) / 4.
    # return psfs_downsample

def copy_quadruple(x_rd):
    x_ld = torch.flip(x_rd, dims=(-2,))
    x_d = torch.cat([x_ld, x_rd], dim=-2)
    x_u = torch.flip(x_d, dims=(-1,))
    x = torch.cat([x_u, x_d], dim=-1)
    return x

##version1 解卷积后的图像可能有负值
def L2_deconvolution(captimgs, psf_center, gamma):
    # deconvolution method1

    # psf_center *= (captimgs.shape[-2]*captimgs.shape[-1])
    # captimgs_norm = captimgs / torch.sum(captimgs, dim=[-2, -1], keepdim=True) * (captimgs.shape[-2]*captimgs.shape[-1])

    psf_center_fft = fft2(ifftshift(psf_center, dim=[-2, -1]))
    HtH = psf_center_fft * psf_center_fft.conj()
    denominator = HtH + gamma

    captimgs_fft = fft2(captimgs)
    numerator = captimgs_fft * psf_center_fft.conj()
    outputs = torch.real(ifft2(numerator / denominator))
    outputs[outputs<0] = 0
    # scale = captimgs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] / \
    #         outputs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    #
    # # scale = torch.sum(captimgs, dim=[-2, -1], keepdim=True) / torch.sum(outputs, dim=[-2, -1], keepdim=True)
    # outputs *= scale  # 这里让两者总能量相等是不是更好一些

    return outputs

def save_tensor_as_gray(tensor: torch.Tensor, path: str):
    """
    将 [H, W] 的归一化 tensor 保存为灰度图 (8-bit)。
    假设 tensor.sum() == 1.
    """
    # 转到 CPU 并转 numpy
    arr = tensor.detach().cpu().numpy()

    # 归一化到 [0,255]
    arr = arr / arr.max() * 255.0
    arr = arr.astype('uint8')

    # 保存为灰度图
    img = Image.fromarray(arr, mode="L")
    img.save(path)
