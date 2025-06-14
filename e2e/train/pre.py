import logging
import typing

import dnois
import lightning
import torch

import e2e.data
import e2e.train
from e2e.model import ImagingSystem
from e2e.specification import parse_spec_file, FromSpecification, Template
from e2e.train import common

__all__ = [
    'main',

    'PreTrainingFramework',
]

logger = logging.getLogger(__name__)


class PreTrainingFramework(common.CITrainingBase, FromSpecification):
    psf_cache: ...

    def __init__(
        self,
        imaging_system_model: ImagingSystem,
        nn_lr: float,
        lr_decay_factor: float,
        lr_decay_interval: int,
        warmup_steps: int,
        log_image_interval: int,
    ):
        super().__init__(imaging_system_model)
        self.model.o.freeze()
        logger.info('All parameters in lens are frozen')

        self.nn_lr = nn_lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_interval = lr_decay_interval
        self.warmup_steps = warmup_steps
        self.log_image_interval = log_image_interval

        self.register_buffer('psf_cache', None, False)

    def forward(self, image, **kwargs):
        if self.psf_cache is not None:
            return self.model(image, **kwargs, _psf_cache=self.psf_cache)

        container = dnois.utils.VarDict()
        self.model.o.register_variable_hook('patchwise_render.psf', container.collector('psf'))

        result = self.model(image, **kwargs)

        self.psf_cache = container['psf']
        self.model.o.remove_variable_hook('patchwise_render.psf')
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.nn.parameters(), self.nn_lr)
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
        loss = self.train_loss(pred, gt)

        self.log('train/loss', loss)
        if self.global_step % self.log_image_interval == 0:
            self.log_image('train/gt', gt)
            self.log_image('train/pred', pred)
            self.log_image('train/captured', captured)
        return loss

    def validation_step(self, batch, batch_idx):
        gt = batch['image']
        pred, captured = self(gt)
        for k, m in self.val_metrics.items():
            self.log(f'val/{k}', m(pred, gt.contiguous()))

    def warmup(self, optimizer):
        if not (self.warmup_steps > 0 and self.global_step <= self.warmup_steps):
            return
        lr_scale = min(1., self.global_step / self.warmup_steps)
        optimizer.param_groups[0]['lr'] = lr_scale * self.nn_lr

    def log_image(self, label, image):
        self.logger.experiment.add_images(label, image[0], self.global_step, dataformats='CHW')  # noqa

    @classmethod
    def from_specification(cls, spec: Template) -> typing.Self:
        imaging_system = ImagingSystem.from_specification(spec)
        obj = cls(
            imaging_system,
            spec.nn_lr,
            spec.lr_decay_factor,
            spec.lr_decay_interval,
            spec.warmup_steps,
            spec.log_image_interval,
        )

        logger.info('Pre-training framework created')
        return obj


def get_args():
    parser = common.get_base_arg_parser()

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    spec = parse_spec_file(args.spec_file)

    lightning.seed_everything(spec.random_seed)
    torch.set_float32_matmul_precision('high')

    logging.basicConfig(level=logging.INFO)

    m = PreTrainingFramework.from_specification(spec)
    trainer = e2e.train.create_trainer_from_spec(spec)
    dm = e2e.data.HyperSimDM.from_specification(spec)

    logger.info('Preparation completed, starting pre-training...')
    trainer.fit(m, datamodule=dm)


if __name__ == '__main__':
    main()
