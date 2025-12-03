import torch
import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning.callbacks import LearningRateMonitor
try:
    from pytorch_lightning.loggers import WandbLogger
    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    WandbLogger = None
    _WANDB_AVAILABLE = False

from torch.utils.data import ConcatDataset, DataLoader, Subset
from dataset import *

from util.helper import *
from util.log_manager import LogManager
from lightning_flatscope import flatscope

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

seed_everything(3407)

def prepare_data(hparams):  # 把训练集划分为18k训练和4k验证
    image_sz = hparams.image_size
    dataset = hparams.dataset
    simdata_dir = hparams.simdata_dir
    augment = hparams.augment

    padding = 0
    val_idx = 50
    if not augment:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.CenterCrop(image_sz)])
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.CenterCrop(image_sz),
            transforms.ToTensor()
        ])
    dataset = fluorescence(data_folder=simdata_dir, input_transforms=transform)

    train_dataset = Subset(dataset, range(val_idx, len(dataset)))
    val_dataset = Subset(dataset, range(val_idx))

    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_sz,
                                  num_workers=hparams.num_workers, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_sz,
                                num_workers=hparams.num_workers, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader


def main(args):
    loggers = []
    log_dir = os.path.join(args.default_root_dir, args.experiment_name)

    if args.use_tensorboard:
        tb_logger = TensorBoardLogger(args.default_root_dir, name=args.experiment_name)
        loggers.append(tb_logger)
        log_dir = tb_logger.log_dir

    if args.use_wandb:
        if not _WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Please install wandb or disable --use_wandb")
        assert WandbLogger is not None
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name or args.experiment_name,
            save_dir=args.default_root_dir,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
        )
        wandb_logger.experiment.config.update(vars(args), allow_val_change=True)
        loggers.append(wandb_logger)

    if loggers:
        logger = loggers[0] if len(loggers) == 1 else LoggerCollection(loggers)
    else:
        logger = False
        os.makedirs(log_dir, exist_ok=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logmanager_callback = LogManager(fallback_log_dir=log_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        filename='{epoch}-{validation/image_loss:.7f}',
        verbose=True,
        # monitor='val_loss',
        monitor='validation/image_loss',
        save_last=True,
        save_top_k=3,
        mode='min',
    )

    system = flatscope(hparams=args, log_dir=log_dir)
    train_dataloader, val_dataloader = prepare_data(hparams=args)
    
    # 修改这部分
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[logmanager_callback, lr_monitor, checkpoint_callback],
        sync_batchnorm=False,
        benchmark=True,
        accelerator="gpu",
        # devices=1,  # 注释掉这行,使用命令行的 --gpus 参数
        detect_anomaly=True,
    )

    trainer.fit(system, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser.add_argument('--experiment_name', type=str, default='Learned_flatscope')
    parser.add_argument('--dataset', type=str, default='fluorescence')
    parser.add_argument('--mix_dataset', action='store_true')
    parser.add_argument('--simdata_dir', type=str, default=r"D:/user_doc/Remote/DOE/data/sample_data_500_img_crop")
    parser.set_defaults(mix_dataset=False)

    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='flatscope', help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Optional custom WandB run name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Optional WandB entity/org')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases run mode')
    parser.add_argument('--no_tensorboard', dest='use_tensorboard', action='store_false',
                        help='Disable TensorBoard logging if you only want WandB')
    parser.set_defaults(use_tensorboard=True)

    parser = Trainer.add_argparse_args(parser)
    parser = flatscope.add_model_specific_args(parser)

    parser.set_defaults(
        gpus=1,
        default_root_dir=r"D:/user_doc/Remote/DOE/end2end_framework/training_logs",   #整个训练文件所在的根目录，自行修改
        max_epochs=500,
        precision=32,
    )

    args = parser.parse_args()

    main(args)