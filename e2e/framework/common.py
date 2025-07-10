import logging
import typing

import dnois
import lightning
import torch
import torchmetrics as tm

from e2e.model import ImagingSystem
from e2e.specification import FromSpecification, Template

__all__ = [
    'create_train_loss',
    'create_val_metrics',

    'CIFramework',
]

logger = logging.getLogger(__name__)
FrameworkAction = typing.Callable[['CIFramework', Template], None]


def create_val_metrics():
    return torch.nn.ModuleDict({
        'mae': tm.MeanAbsoluteError(),
        'psnr': tm.image.PeakSignalNoiseRatio(1.),  # PSNR需要设定最大值为1
        'ssim': tm.image.StructuralSimilarityIndexMeasure(),
    })


def create_train_loss():
    return tm.MeanAbsoluteError()


class CIFramework(lightning.LightningModule, FromSpecification):
    name: str
    execute_train: FrameworkAction

    def __init__(
        self,
        imaging_system_model: ImagingSystem,
    ):
        super().__init__()
        self.model = imaging_system_model

        self.train_loss = create_train_loss()

        self.val_metrics = create_val_metrics()

    def __getattr__(self, item):
        if item.startswith('execute_'):
            raise NotImplementedError(f'Action {item[8:]} is not implemented for framework {self.name}')
        return super().__getattr__(item)

    def execute(self, action: str, spec: Template):
        method_name = f'execute_{action}'
        method = getattr(self, method_name)
        return method(spec)

    @classmethod
    def from_specification(cls, spec: Template) -> typing.Self:
        if cls is not CIFramework:
            raise NotImplementedError(f'{cls.from_specification.__qualname__} is not implemented')

        subclasses = dnois.utils.subclasses(CIFramework)
        for subclass in subclasses:
            if subclass.name == spec.framework:
                logger.info(f'Framework detected: {spec.framework} (class {subclass.__name__})')
                return subclass.from_specification(spec)
        available = [subclass.name for subclass in subclasses]
        raise ValueError(f'Framework {spec.framework} is not found, available: {", ".join(available)}')
