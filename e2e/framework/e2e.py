import logging
import typing

import dnois
import torch

import e2e.data
from e2e.model import ImagingSystem
from e2e.specification import FromSpecification, Template

from .common import *
from .trainer import *

__all__ = [
    'TrainingFramework',
]

logger = logging.getLogger(__name__)


class TrainingFramework(CIFramework, FromSpecification):
    name = 'e2e'

    def __init__(
        self,
        imaging_system_model: ImagingSystem,
        optics_lr: float,
        nn_lr: float,
        lr_decay_factor: float,
        lr_decay_interval: int,
        warmup_steps: int,
        target_fl: float,
        fl_loss_w: float,
        log_image_interval: int,
    ):
        super().__init__(imaging_system_model)

        self.optics_lr = optics_lr
        self.nn_lr = nn_lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_interval = lr_decay_interval
        self.warmup_steps = warmup_steps
        self.target_fl = target_fl
        self.fl_loss_w = fl_loss_w
        self.log_image_interval = log_image_interval

    def forward(self, image, **kwargs):
        result = self.model(image, **kwargs)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.o.parameters(), 'lr': self.optics_lr},
            {'params': self.model.nn.parameters(), 'lr': self.nn_lr},
        ])
        logger.info('Optimizer configured')
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay_factor),
                'interval': 'epoch',
                'frequency': self.lr_decay_interval,
            },
        }

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure=None) -> None:
        self.warmup(optimizer)
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def training_step(self, batch, batch_idx):
        gt = batch['image']
        pred, captured = self(gt)
        restoration_loss = self.train_loss(pred, gt)

        fl_loss = self.fl_loss()
        loss = restoration_loss + self.fl_loss_w * fl_loss

        self.log('train/restoration_loss', restoration_loss)
        self.log('train/fl_loss', fl_loss)
        self.log_param()
        if self.global_step % self.log_image_interval == 0:
            self.log_image('train/gt', gt)
            self.log_image('train/pred', pred)
            self.log_image('train/captured', captured)
            sw = self.logger.experiment  # noqa
            sw.add_image('train/psf', self.make_psf(), self.global_step, dataformats='CHW')
            sw.add_figure('train/lens', self.make_figure(), self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        gt = batch['image']
        pred, captured = self(gt)
        for k, m in self.val_metrics.items():
            self.log(f'val/{k}', m(pred, gt.contiguous()))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def warmup(self, optimizer):
        if not (self.warmup_steps > 0 and self.global_step <= self.warmup_steps):
            return
        lr_scale = min(1., self.global_step / self.warmup_steps)
        optimizer.param_groups[0]['lr'] = lr_scale * self.nn_lr

    def fl_loss(self):
        fl = self.model.o.focal_length2(wl_reduction='center')  # 只取中心波长的焦距
        fl_loss = (fl - self.target_fl).square()
        return fl_loss

    def log_param(self):
        sq = self.model.o.surfaces
        self.log(f'param/2/roc', sq[2].roc)
        for i in range(2, 5):
            self.log(f'param/2/a{i}', getattr(sq[2], f'a{i}'))
            self.log(f'param/3/a{i}', getattr(sq[3], f'a{i}'))
        self.log(f'param/4/distance', sq[4].distance)

    @torch.no_grad()
    def make_psf(self):
        o = self.model.o
        # 通过points_grid方法创建一个物点的二维阵列（根据视场范围、分块数和物距）
        obj_points = o.points_grid(o.segments, o.depth.squeeze().item())
        # 只取1/4的物点（根据对称性）
        obj_points = obj_points[..., obj_points.size(-3) // 2:, obj_points.size(-2) // 2:, :]
        # 计算PSF中心时，每个波长独立计算（而非取平均或中心波长）
        psf = o.psf(obj_points, wl_reduction='none')
        # 为了方便观察，除以最大值并开平方
        psf /= psf.max()
        psf = psf.sqrt()
        psf = dnois.utils.resize(psf, 32)
        psf = torch.cat([
            torch.cat([psf[i, j] for j in range(psf.size(1))], -1)
            for i in range(psf.size(0))
        ], -2)  # N_wl x H x W
        return psf

    @torch.no_grad()
    def make_figure(self):
        return self.model.o.plot_cross_section()

    def log_image(self, label, image):
        self.logger.experiment.add_images(label, image[0], self.global_step, dataformats='CHW')  # noqa

    def load_nn_pretrained(self, nn_init_path):
        ckpt = torch.load(nn_init_path, weights_only=True)
        prefix = 'model.nn.'
        nn_sd = {k[len(prefix):]: v for k, v in ckpt['state_dict'].items() if k.startswith(prefix)}
        self.model.nn.load_state_dict(nn_sd)

    def execute_train(self, spec: Template, tensorboard_on: bool = True):
        trainer = create_trainer_from_spec(spec)
        dm = e2e.data.HyperSimDM.from_specification(spec)

        logger.info('Preparation completed, starting end-to-end training...')
        cm = train_context_manager(spec, trainer, tensorboard_on)
        with cm:
            trainer.fit(self, datamodule=dm)

    def execute_eval(self, spec: Template):
        trainer = create_trainer_from_spec(spec)
        trainer.logger = None
        dm = e2e.data.HyperSimDM.from_specification(spec)

        logger.info('Preparation completed, starting evaluation...')
        trainer.test(self, datamodule=dm)

    @classmethod
    def from_specification(cls, spec: Template) -> typing.Self:
        imaging_system = ImagingSystem.from_specification(spec)
        kwargs = {
            'imaging_system_model': imaging_system,
            'optics_lr': spec.optics_lr,
            'nn_lr': spec.nn_lr,
            'lr_decay_factor': spec.lr_decay_factor,
            'lr_decay_interval': spec.lr_decay_interval,
            'warmup_steps': spec.warmup_steps,
            'target_fl': spec.target_focal_length,
            'fl_loss_w': spec.focal_length_loss_weight,
            'log_image_interval': spec.log_image_interval,
        }

        if spec.trained_ckpt_path is None:
            obj = cls(**kwargs)
            if spec.nn_init_path is not None:
                logger.info(f'Loading pretrained NN from {spec.nn_init_path}')
                obj.load_nn_pretrained(spec.nn_init_path)
        else:
            obj = cls.load_from_checkpoint(spec.trained_ckpt_path, **kwargs)  # noqa

        logger.info('End-to-end training framework created')
        return obj
