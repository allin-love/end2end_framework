from argparse import ArgumentParser, Namespace
from typing import cast
import torch
import torch.optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.utils
from camera import camera_binary_rings
from util.helper import *


class flatscope(pl.LightningModule):
    def __init__(self, hparams, log_dir=None):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.__build_model()
        self.log_dir = log_dir

    @property
    def cfg(self) -> Namespace:
        return cast(Namespace, self.hparams)

    def configure_optimizers(self):
        params = [{'params': self.camera.parameters(), 'lr': self.cfg.optics_lr}]
        optimizer = torch.optim.Adam(params)
        lr_gamma = getattr(self.cfg, 'lr_gamma', 0.98)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma),
                        'name': 'lr_decay_curve'}
        return [optimizer], [lr_scheduler]


    def training_step(self, samples, batch_idx):
        psfs, targets, captimgs, outputs = self.forward(samples)
        data_loss, loss_logs = self.__compute_loss(outputs, targets, psfs)
        loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}

        if self.cfg.optimize_optics:
            misc_logs = {
                'optics/optim_param_max': self.camera.optim_param.max(),
                'optics/optim_param_min': self.camera.optim_param.min(),
                'optics/optim_param_mean': self.camera.optim_param.mean(),
            }
            with torch.no_grad():
                ring_radii = self.camera.get_ring_radii().detach()
                lens_radius_mm = self.cfg.lens_diameter * 1e3 / 2.0
                spacing = torch.diff(ring_radii, prepend=ring_radii.new_tensor([0.0])) * lens_radius_mm
                misc_logs['optics/min_ring_spacing_um'] = spacing.min() * 1e3
                misc_logs['optics/max_ring_spacing_um'] = spacing.max() * 1e3
                misc_logs['optics/mask_pixel_pitch_um'] = torch.tensor(self.camera.mask_pixel_pitch * 1e6)

        logs = {}
        logs.update(loss_logs)
        if self.cfg.optimize_optics:
            logs.update(misc_logs)
        self.log_dict(logs)

        if not self.global_step % self.cfg.summary_track_train_every:
            self.__log_images(psfs, targets, captimgs, outputs, 'train')

        return data_loss

    def validation_step(self, samples, batch_idx):
        psfs, targets, captimgs, outputs = self.forward(samples)

        repeated_targets = targets.repeat(1, outputs.shape[1], 1, 1)
        val_loss = F.mse_loss(outputs.contiguous(), repeated_targets.contiguous())
        self.log('validation/image_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self.__log_images(psfs, targets, captimgs, outputs, 'validation')

    def forward(self, targets):
        psfs = self.camera.gen_psf()

        bin_factor = getattr(self.cfg, 'psf_binning_factor', 1)
        if bin_factor and bin_factor > 1:
            psfs = binnning(bin_factor, psfs).float()
        else:
            psfs = psfs.float()

        target_size = self.cfg.image_size
        psf_size = psfs.shape[-1]
        if psf_size < target_size:
            padding = (target_size - psf_size) // 2
            psfs_for_conv = F.pad(psfs, (padding, padding, padding, padding))
        elif psf_size > target_size:
            psfs_for_conv = crop_image(psfs, target_size)
        else:
            psfs_for_conv = psfs

        noise_sigma = (self.cfg.noise_sigma_max - self.cfg.noise_sigma_min) * torch.rand(
            (targets.shape[0], 1, 1, 1), device=targets.device,
            dtype=targets.dtype) + self.cfg.noise_sigma_min
        chunk_size = getattr(self.cfg, 'psf_chunk_size', 0)
        captimgs = conv_psf(targets, psfs_for_conv, chunk_size=chunk_size) + \
                   noise_sigma * torch.randn((targets.shape[0], 1, targets.shape[-2], targets.shape[-1]),
                                             device=targets.device, dtype=targets.dtype)

        recon_crop = getattr(self.cfg, 'reconstruction_crop', getattr(self.cfg, 'psf_crop_size', self.cfg.image_size))
        captimgs = crop_image(captimgs, recon_crop)
        outputs = captimgs
        targets = crop_image(targets, recon_crop)
        return psfs, targets, captimgs, outputs

    def __build_model(self):   #框架各部分模块配置
        hparams = self.cfg

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
        camera_recipe_rings['window_cropsize'] = getattr(hparams, 'psf_window', camera_recipe_rings['image_size'])
        camera_recipe_rings['mask_pixel_pitch'] = getattr(hparams, 'mask_pixel_pitch', 2e-6)
        camera_recipe_rings['ring_softness'] = getattr(hparams, 'ring_softness', 60.0)
        camera_recipe_rings['psf_crop_size'] = getattr(hparams, 'psf_crop_size', camera_recipe_rings['window_cropsize'])
        self.camera = camera_binary_rings.BinaryRingsCamera(**camera_recipe_rings, require_grad=optimize_optics)

        depth_min = getattr(hparams, 'depth_min', -0.5e-3)
        depth_max = getattr(hparams, 'depth_max', 0.5e-3)
        depth_planes = max(2, getattr(hparams, 'depth_planes', 5))
        self.camera.delta_z = torch.linspace(depth_min, depth_max, depth_planes).tolist()

        print(self.camera)

    def __compute_loss(self, outputs, targets, psfs=None):

        image_l1_loss = self.img_l1_loss(targets, outputs)

        logs = {
            'image_l1_loss': image_l1_loss,
        }

        total_loss = self.cfg.l1_loss_weight * image_l1_loss

        if psfs is not None:
            psf_terms = self.__psf_regularizers(psfs)
            var_weight = getattr(self.cfg, 'psf_consistency_weight', 0.0)
            worst_weight = getattr(self.cfg, 'psf_worst_weight', 0.0)
            focus_weight = getattr(self.cfg, 'focus_balance_weight', 0.0)
            if var_weight > 0:
                total_loss = total_loss + var_weight * psf_terms['variance']
                logs['psf/variance'] = psf_terms['variance'].detach()
            if worst_weight > 0:
                total_loss = total_loss + worst_weight * psf_terms['worst_l1']
                logs['psf/worst_l1'] = psf_terms['worst_l1'].detach()

            target_center = getattr(self.cfg, 'focus_center_target', -1.0)
            if target_center < 0:
                target_center = (psfs.shape[1] - 1) / 2.0
            focus_penalty = torch.mean((psf_terms['depth_center'] - target_center) ** 2)
            if focus_weight > 0:
                total_loss = total_loss + focus_weight * focus_penalty
                logs['psf/focus_shift_penalty'] = focus_penalty.detach()

            logs.setdefault('psf/variance', psf_terms['variance'].detach())
            logs.setdefault('psf/worst_l1', psf_terms['worst_l1'].detach())
            logs['psf/depth_center'] = psf_terms['depth_center'].mean().detach()

        return total_loss, logs


    def img_l1_loss(self, targets, outputs):
        return F.l1_loss(outputs, targets)

    def __psf_regularizers(self, psfs):
        eps = 1e-8
        psfs = psfs / (psfs.sum(dim=(-2, -1), keepdim=True) + eps)
        batch, depth = psfs.shape[0], psfs.shape[1]
        psfs_flat = psfs.view(batch, depth, -1)
        mean_psf = psfs_flat.mean(dim=1, keepdim=True)
        variance = torch.mean((psfs_flat - mean_psf) ** 2)
        per_depth_l1 = torch.mean(torch.abs(psfs_flat - mean_psf), dim=-1)
        worst_l1 = per_depth_l1.max()

        depth_idx = torch.arange(depth, device=psfs.device, dtype=psfs.dtype)
        energy = psfs_flat.sum(dim=-1)
        depth_prob = energy / (energy.sum(dim=1, keepdim=True) + eps)
        depth_center = (depth_prob * depth_idx).sum(dim=1)

        return {
            'variance': variance,
            'worst_l1': worst_l1,
            'depth_center': depth_center,
        }

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

        if self.logger is not None and getattr(self.logger, 'experiment', None) is not None:
            if tag == 'test':
                self.logger.experiment.add_image(f'{tag}/summary_image_{batch_idx}', grid_summary_image, self.global_step)
            else:
                self.logger.experiment.add_image(f'{tag}/summary_image', grid_summary_image, self.global_step)

        if (self.cfg.optimize_optics or self.global_step == 0) and self.logger is not None and getattr(self.logger, 'experiment', None) is not None:
            phase_bias = imresize(self.camera.phase_bias() % (2 * torch.pi), self.cfg.summary_image_sz)
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
        parser.add_argument('--optics_lr', type=float, default=2e-3)
        parser.add_argument('--lr_gamma', type=float, default=0.98)
        parser.add_argument('--batch_sz', type=int, default=1)
        parser.add_argument('--test_batch_sz', type=int, default=8)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--l1_loss_weight', type=float, default=1.0)
        parser.add_argument('--reconstruction_crop', type=int, default=288)
        parser.add_argument('--psf_binning_factor', type=int, default=1)
        parser.add_argument('--psf_chunk_size', type=int, default=2)
        parser.add_argument('--psf_consistency_weight', type=float, default=0.0)
        parser.add_argument('--psf_worst_weight', type=float, default=0.0)
        parser.add_argument('--focus_balance_weight', type=float, default=0.0)
        parser.add_argument('--focus_center_target', type=float, default=-1.0)

        parser.add_argument('--noise_sigma_min', type=float, default=0.0)
        parser.add_argument('--noise_sigma_max', type=float, default=0.01)
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
        parser.add_argument('--psf_window', type=int, default=288)
        parser.add_argument('--mask_pixel_pitch', type=float, default=2e-6)
        parser.add_argument('--ring_softness', type=float, default=80.0)
        parser.add_argument('--psf_crop_size', type=int, default=288,
                    help='Central PSF region (pixels) retained after propagation; set smaller than psf_window to shrink FOV without altering sampling.')
        parser.add_argument('--depth_min', type=float, default=-0.5e-3)
        parser.add_argument('--depth_max', type=float, default=0.5e-3)
        parser.add_argument('--depth_planes', type=int, default=3)
        parser.add_argument('--optimize_optics', dest='optimize_optics', action='store_true')
        parser.add_argument('--no-optimize_optics', dest='optimize_optics', action='store_false')
        parser.set_defaults(optimize_optics=True)

        return parser