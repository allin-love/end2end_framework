import torch
import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from torch.utils.data import ConcatDataset, DataLoader
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

    train_dataset = torch.utils.data.Subset(dataset,
                                            range(val_idx, len(dataset)))
    val_dataset = torch.utils.data.Subset(dataset, range(val_idx))

    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_sz,
                                  num_workers=hparams.num_workers, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_sz,
                                num_workers=hparams.num_workers, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader


def main(args):
    logger = TensorBoardLogger(args.default_root_dir,
                               name=args.experiment_name)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    logmanager_callback = LogManager()

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
        filename='{epoch}-{validation/image_loss:.7f}',
        verbose=True,
        # monitor='val_loss',
        monitor='validation/image_loss',
        save_last=True,
        save_top_k=3,
        mode='min',
    )

    system = flatscope(hparams=args, log_dir=logger.log_dir)
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
    parser.add_argument('--simdata_dir', type=str, default=r"D:/user_doc/Remote/DOE/data/sample_data_500_mag0.4_img_crop")
    parser.set_defaults(mix_dataset=False)

    parser = Trainer.add_argparse_args(parser)
    parser = flatscope.add_model_specific_args(parser)

    parser.set_defaults(
        gpus=1,
        default_root_dir=r"D:/user_doc/Remote/DOE/end2end_framework/training_logs",   #整个训练文件所在的根目录，自行修改
        max_epochs=50,
    )

    args = parser.parse_args()

    with torch.autograd.set_detect_anomaly(True):
        main(args)