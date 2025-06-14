import argparse

import lightning
import torch
import torchmetrics as tm

from e2e.model import ImagingSystem

__all__ = [
    'create_train_loss',
    'create_val_metrics',

    'CITrainingBase',
]


def create_val_metrics():
    return torch.nn.ModuleDict({
        'mae': tm.MeanAbsoluteError(),
        'psnr': tm.image.PeakSignalNoiseRatio(1.),  # PSNR需要设定最大值为1
        'ssim': tm.image.StructuralSimilarityIndexMeasure(),
    })


def create_train_loss():
    return tm.MeanAbsoluteError()


def get_base_arg_parser():
    parser = argparse.ArgumentParser(
        description='End-to-end optimization of computation imaging system'
    )

    parser.add_argument('spec_file', type=str)
    return parser


class CITrainingBase(lightning.LightningModule):
    def __init__(
        self,
        imaging_system_model: ImagingSystem,
    ):
        super().__init__()
        self.model = imaging_system_model

        self.train_loss = create_train_loss()

        self.val_metrics = create_val_metrics()
