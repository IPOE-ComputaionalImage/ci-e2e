import logging
import typing

import dnois
import torch

import viflo.data
from viflo.model import ImagingSystem
from viflo.specification import FromSpecification, Template

from .core import *
from .trainer import *

__all__ = [
    'Deblur',
]

logger = logging.getLogger(__name__)


class Deblur(CIFramework, FromSpecification):
    name = 'deblur'

    def __init__(
        self,
        imaging_system_model: ImagingSystem,
        spec: Template,
    ):
        super().__init__(imaging_system_model)
        self.spec = spec

    def forward(self, image, **kwargs):
        result = self.model(image, **kwargs)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.o.parameters(), 'lr': self.spec.optics_lr},
            {'params': self.model.nn.parameters(), 'lr': self.spec.nn_lr},
        ])
        logger.info('Optimizer configured')
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, self.spec.lr_decay_factor),
                'interval': 'epoch',
                'frequency': self.spec.lr_decay_interval,
            },
        }

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure=None) -> None:
        self.warmup(optimizer)
        with self.spec.optimizer_step_context():
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def training_step(self, batch, batch_idx):
        losses = []

        gt = batch['image']
        kwargs = self.spec.keywords_to_optical_model or {}
        kwargs = kwargs.get('train', {})
        pred, captured, result = self(gt, **kwargs)
        gt = self.model.o.crop(gt)

        restoration_loss = self.train_loss(pred, gt.contiguous())
        self.log('train/restoration_loss', restoration_loss)
        losses.append(restoration_loss)

        if self.spec.focal_length_loss_weight is not None:
            fl_loss = self.fl_loss()
            self.log('train/fl_loss', fl_loss)
            losses.append(self.spec.focal_length_loss_weight * fl_loss)

        loss = sum(losses)

        self.log_param()
        if self.global_step % self.spec.log_image_interval == 0:
            self.log_image('train/gt', gt)
            self.log_image('train/pred', pred)
            self.log_image('train/captured', captured)
            sw = self.logger.experiment  # noqa
            sw.add_image('train/psf', self.make_psf(), self.global_step, dataformats='CHW')
            sw.add_figure('train/lens', self.make_figure(), self.global_step)

        self.handle_raw_output(result)
        return loss

    def validation_step(self, batch, batch_idx):
        gt = batch['image']

        kwargs = self.spec.keywords_to_optical_model or {}
        kwargs = kwargs.get('val', {})
        pred, captured, _ = self(gt, **kwargs)
        gt = self.model.o.crop(gt)

        for k, m in self.val_metrics.items():
            self.log(f'val/{k}', m(pred, gt.contiguous()))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def warmup(self, optimizer):
        warmup_steps = self.spec.warmup_steps
        if not (warmup_steps > 0 and self.global_step <= warmup_steps):
            return

        lr_scale = min(1., self.global_step / warmup_steps)
        optimizer.param_groups[0]['lr'] = lr_scale * self.spec.optics_lr
        optimizer.param_groups[1]['lr'] = lr_scale * self.spec.nn_lr

    def handle_raw_output(self, result):
        pass

    def fl_loss(self):
        kwargs = {'wl_reduction': 'center'}
        if self.spec.focal_length_type is not None:
            kwargs['fl_type'] = self.spec.focal_length_type
        fl = self.model.o.focal_length2(**kwargs)

        fl_loss = (fl - self.spec.target_focal_length).square()
        return fl_loss

    def log_param(self):
        for i, s in enumerate(self.model.o.surfaces):
            for k, p in s.nominal_values.items():
                if p.requires_grad:
                    self.log(f'param/{i}/{k}', p)
            for k, p in s.context.nominal_values.items():
                if p.requires_grad:
                    self.log(f'param/{i}/{k}', p)

    @torch.no_grad()
    def make_psf(self):
        o = self.model.o
        obj_points = o.points_grid(self.spec.visualize_psf_cells, o.depth.squeeze().item())
        if self.spec.visualize_psf_quadrant:
            obj_points = obj_points[..., obj_points.size(-3) // 2:, obj_points.size(-2) // 2:, :]
        psf = o.psf(obj_points, wl_reduction='none')
        psf /= psf.max()
        psf = psf.sqrt()
        psf = dnois.utils.resize(psf, self.spec.visualize_psf_cell_size)
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

    def execute_train(self, spec: Template, tensorboard_on: bool = True):
        init_ckpt_path = spec.initializing_checkpoint_path
        if init_ckpt_path is not None:
            logger.info(f'Loading state_dict from {init_ckpt_path} for initialization')
            init_ckpt = torch.load(init_ckpt_path, weights_only=True)
            self.load_state_dict(init_ckpt['state_dict'])

        trainer = create_trainer_from_spec(spec)
        dm = viflo.data.data_module_from_spec(spec, 'train')

        logger.info('Preparation completed, starting end-to-end training...')
        cm = train_context_manager(spec, trainer, tensorboard_on)
        with cm:
            trainer.fit(self, datamodule=dm)

    def execute_pretrain(self, spec: Template, tensorboard_on: bool = True):
        self.model.o.freeze()
        logger.info('All the optical parameters are frozen')

        trainer = create_trainer_from_spec(spec)
        dm = viflo.data.data_module_from_spec(spec, 'pretrain')

        logger.info('Preparation completed, starting NN pre-training...')
        cm = train_context_manager(spec, trainer, tensorboard_on)
        with cm:
            trainer.fit(self, datamodule=dm)

    def execute_eval(self, spec: Template):
        trainer = create_trainer_from_spec(spec)
        trainer.logger = None
        dm = viflo.data.data_module_from_spec(spec, 'eval')

        logger.info('Preparation completed, starting evaluation...')
        trainer.test(self, datamodule=dm)

    @classmethod
    def from_specification(cls, spec: Template) -> typing.Self:
        if spec.trained_checkpoint_path is None:
            logger.info('No checkpoint path specified, creating a new model')
            imaging_system = ImagingSystem.from_specification(spec)
            obj = cls(imaging_system, spec)
        else:
            logger.info(f'Loading checkpoint from {spec.trained_checkpoint_path}')
            obj = cls.load_from_checkpoint(spec.trained_ckpt_path, spec=spec)  # noqa

        logger.info(f'Framework {cls.__name__} created')
        return obj
