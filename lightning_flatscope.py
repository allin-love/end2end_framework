import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
from network.simple_model import SimpleModel
from argparse import ArgumentParser
import torchvision.utils
from camera import camera_zernike_axial, camera_rotation_axial, camera_binary_rings
from torchmetrics.regression import *
from torchmetrics.image import PeakSignalNoiseRatio
from util.helper import *
from util.deconvolution import apply_tikhonov_inverse
from torch.fft import ifftshift, ifft2


class flatscope(pl.core.LightningModule):
    def __init__(self, hparams, log_dir=None):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.__build_model()
        self.metrics = nn.ModuleDict({'image_loss': MeanSquaredError()})

        self.evaluation_metrics = nn.ModuleDict({
            # 'image_loss': MeanSquaredError(),
            'image_loss': MeanAbsoluteError(),
            'image_psnr': PeakSignalNoiseRatio(),
        })

        self.log_dir = log_dir

    def configure_optimizers(self):
        if self.hparams.deconvolution:   #在重建部分使用解卷机
            params = [
                {'params': self.camera.parameters(), 'lr': self.hparams.optics_lr},
                {'params': self.gamma, 'lr': self.hparams.gamma_lr},
            ]
        else:  #在重建部分使用神经网络
            params = [
                {'params': self.camera.parameters(), 'lr': self.hparams.optics_lr},
                {'params': self.decoder.parameters(), 'lr': self.hparams.cnn_lr},
            ]
        optimizer = torch.optim.Adam(params)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.gamma),
                        'name': 'lr_decay_curve'}
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        # # warm up lr
        # if self.trainer.global_step < 4000:
        #     lr_scale = min(1., float(self.trainer.global_step + 1) / 4000.)
        #     optimizer.param_groups[0]['lr'] = lr_scale * self.hparams.optics_lr
        #     optimizer.param_groups[1]['lr'] = lr_scale * self.hparams.cnn_lr

        # update params
        optimizer.step(closure=optimizer_closure)  # 在用高版本的pytorch-lightning时，要把optimizer_closure传入optimizer.step()
        optimizer.zero_grad()

    def training_step(self, samples, batch_idx):
        psfs, targets, captimgs, outputs = self.forward(samples)
        data_loss, loss_logs = self.__compute_loss(outputs, targets)
        loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}

        if self.hparams.optimize_optics:
            misc_logs = {
                'optics/optim_param_max': self.camera.optim_param.max(),
                'optics/optim_param_min': self.camera.optim_param.min(),
                'optics/optim_param_mean': self.camera.optim_param.mean(),
                'optics/L2_gamma_mean': self.gamma.mean(),
                'optics/L2_gamma_min': self.gamma.min(),
                'optics/L2_gamma_max': self.gamma.max(),
            }

        logs = {}
        logs.update(loss_logs)
        if self.hparams.optimize_optics:
            logs.update(misc_logs)
        self.log_dict(logs)

        if not self.global_step % self.hparams.summary_track_train_every:
            self.__log_images(psfs, targets, captimgs, outputs, 'train')

        return data_loss

    def on_validation_epoch_start(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
            metric.to(self.device)

    def validation_step(self, samples, batch_idx):
        psfs, targets, captimgs, outputs = self.forward(samples)


        # 这是原始代码 (Line 96)
        # self.metrics['image_loss'](outputs, targets.repeat(1, outputs.shape[1],1,1))

        self.metrics['image_loss'](outputs.contiguous(), targets.repeat(1, outputs.shape[1],1,1).contiguous())
        self.log('validation/image_loss', self.metrics['image_loss'], on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.__log_images(psfs, targets, captimgs, outputs, 'validation')

    def forward(self, targets):
        psfs = self.camera.gen_psf()

        #####这里光学仿真的采样间隔为0.5um，但是考虑相机实际像元为1um或更大，故对psf仿真结果进行binning，以模拟相机拍到的psf
        # if self.hparams.camera_pixel_pitch < 1e-6:
        #     psfs_downsample = binnning(2, psfs).float()
        # else:
        #     psfs_downsample = psfs.float()
        psfs_downsample = psfs.float() #直接使用0.5um采样间隔的psf进行仿真

        padding = (self.hparams.image_size - psfs_downsample.shape[-1]) // 2
        psfs_downsample = F.pad(psfs_downsample, (padding, padding, padding, padding))
        noise_sigma = (self.hparams.noise_sigma_max - self.hparams.noise_sigma_min) * torch.rand(
            (targets.shape[0], 1, 1, 1), device=targets.device,
            dtype=targets.dtype) + self.hparams.noise_sigma_min
        captimgs = conv_psf(targets, psfs_downsample) + \
                   noise_sigma * torch.randn((targets.shape[0], 1, targets.shape[-2], targets.shape[-1]),
                                             device=targets.device, dtype=targets.dtype)

        if self.hparams.deconvolution:

            center_ind = psfs_downsample.shape[1] // 2
            psf_center = psfs_downsample[:, center_ind].unsqueeze(1)
            gamma = torch.clamp(self.gamma, min=1e-8, max=5e-3)
            outputs = L2_deconvolution(captimgs, psf_center, gamma)
            captimgs = crop_image(captimgs, 288)
            outputs = crop_image(outputs, 288)
        else:
            captimgs = crop_image(captimgs, 288)
            outputs = self.decoder(captimgs)

        targets = crop_image(targets, 288)
        return psfs, targets, captimgs, outputs

    def __build_model(self):   #框架各部分模块配置
        hparams = self.hparams

        camera_recipe = {
            'image_size': hparams.image_size,
            'sensor_diameter': hparams.sensor_diameter,  # 384
            'lens_diameter': hparams.lens_diameter,  # 3.5328e-3
            'camera_pixel_pitch': hparams.camera_pixel_pitch,
            'd1': hparams.d1,
            'd2': hparams.d2,
            'num_polynomials': hparams.num_polynomials,
        }

        optimize_optics = hparams.optimize_optics

        ########选择你需要的cameara（每个cameara的DOE的参数化方式不同）
        # self.camera = camera_zernike_axial.BaseCamera(**camera_recipe, requires_grad=optimize_optics)   #用zernike表示相位面
        # self.camera = camera_rotation_axial.BaseCamera(**camera_recipe, requires_grad=optimize_optics) #用旋转对称型结构表示相位面
        
        # 新增：同心圆二元相位板（更适合加工）
        # 将 num_polynomials 参数映射为 num_rings（环带数量）
        camera_recipe_rings = camera_recipe.copy()
        camera_recipe_rings['num_rings'] = camera_recipe_rings.pop('num_polynomials')  # 用 num_rings 代替 num_polynomials
        self.camera = camera_binary_rings.BinaryRingsCamera(**camera_recipe_rings, require_grad=optimize_optics)

        if not self.hparams.deconvolution:
            self.decoder = SimpleModel(image_ch=len(self.camera.obj_offsets_x) * len(self.camera.obj_offsets_y),
                                       output_ch=len(self.camera.obj_offsets_x) * len(self.camera.obj_offsets_y))
        # self.image_lossfn = nn.MSELoss()

        print(self.camera)
        psf_number = len(self.camera.delta_z) * len(self.camera.obj_offsets_x) * len(self.camera.obj_offsets_y) * len(
            self.camera.wavelengths)
        self.register_parameter('gamma', nn.Parameter(torch.ones(1, psf_number, 1, 1) * 5e-4))

    def __compute_loss(self, outputs, targets):

        image_l1_loss = self.img_l1_loss(targets, outputs)

        logs = {
            'image_l1_loss': image_l1_loss,

        }

        total_loss = self.hparams.l1_loss_weight * image_l1_loss

        return total_loss, logs


    def img_l1_loss(self, targets, outputs):
        
        eps = 1e-8
        targets = targets / (torch.sum(targets, dim=[-2, -1], keepdim=True) + eps)
        outputs = outputs / (torch.sum(outputs, dim=[-2, -1], keepdim=True) + eps)

        diff = outputs - targets
        loss = torch.mean(torch.abs(diff)) * 1e2

        return loss

    @torch.no_grad()
    def __log_images(self, psfs, targets, captimgs, outputs, tag: str, batch_idx=None):


        captimgs_vis = F.relu(captimgs, inplace=False) / \
                       captimgs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        scale = 0.9 / outputs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        outputs_vis = F.relu(outputs, inplace=False) * scale

        summary_image = torch.cat(
            [targets[0].unsqueeze(1).repeat(captimgs.shape[1], 1, 1, 1), captimgs_vis[0].unsqueeze(1),
             outputs_vis[0].unsqueeze(1)], dim=-2)
        grid_summary_image = torchvision.utils.make_grid(summary_image, nrow=captimgs.shape[1], padding=2)

        if tag == 'test':
            self.logger.experiment.add_image(f'{tag}/summary_image_{batch_idx}', grid_summary_image, self.global_step)
        else:
            self.logger.experiment.add_image(f'{tag}/summary_image', grid_summary_image, self.global_step)

        if self.hparams.optimize_optics or self.global_step == 0:
            phase_bias = imresize(self.camera.phase_bias() % (2 * torch.pi), self.hparams.summary_image_sz)
            phase_bias /= phase_bias.max()

            self.logger.experiment.add_image('optics/optim_phase_bias{}'.format(self.global_step), phase_bias[0],
                                             self.global_step)

            psfs /= \
                psfs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            grid_modulated_psfs = torchvision.utils.make_grid(psfs.transpose(0, 1), nrow=3, pad_value=1,
                                                              normalize=False)

            self.logger.experiment.add_image('optics/modulated_psfs_stretched{}'.format(self.global_step),
                                             grid_modulated_psfs, self.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logger parameters
        parser.add_argument('--summary_image_sz', type=int, default=256)
        parser.add_argument('--summary_track_train_every', type=int, default=500)

        # training parameters
        parser.add_argument('--cnn_lr', type=float, default=1e-3)
        parser.add_argument('--optics_lr', type=float, default=2e-3)
        parser.add_argument('--gamma_lr', type=float, default=0e-4)
        parser.add_argument('--batch_sz', type=int, default=1)
        parser.add_argument('--gamma', type=float, default=0.98)
        parser.add_argument('--test_batch_sz', type=int, default=8)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--l1_loss_weight', type=float, default=1.0)

        parser.add_argument('--noise_sigma_min', type=float, default=0.00)
        parser.add_argument('--noise_sigma_max', type=float, default=0.00)
        parser.add_argument('--deconvolution', dest='deconvolution', action='store_true')
        parser.add_argument('--no-deconvolution', dest='deconvolution', action='store_false')
        parser.set_defaults(deconvolution=True)
        parser.add_argument('--augment', default=False, action='store_true')

        # dataset parameters
        parser.add_argument('--image_size', type=int, default=128)

        # optics parameters
        parser.add_argument('--sensor_diameter', type=float, default=5.5e-3)
        parser.add_argument('--camera_pixel_pitch', type=float, default=1.0e-6)
        parser.add_argument('--lens_diameter', type=float, default=3.45e-3)
        parser.add_argument('--d1', type=float, default=65e-3)
        parser.add_argument('--d2', type=float, default=13.59e-3)
        parser.add_argument('--num_polynomials', type=int, default=100)
        parser.add_argument('--optimize_optics', dest='optimize_optics', action='store_true')
        parser.add_argument('--no-optimize_optics', dest='optimize_optics', action='store_false')
        parser.set_defaults(optimize_optics=True)

        return parser