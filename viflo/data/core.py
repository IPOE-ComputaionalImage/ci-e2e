from pathlib import Path
import logging
import typing

import dnois
import lightning
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from ..specification import Template

__all__ = [
    'data_module_from_spec',
    'make_cropper',

    'BasicDataset',
    'VifloDataModule',
    'MixedDataModule',
]
logger = logging.getLogger(__name__)


def data_module_from_spec(spec: Template, action: str) -> lightning.LightningDataModule:
    try:
        return spec.create_data_module(action)
    except NotImplementedError:
        pass

    if spec.data_module is None:
        raise ValueError(f'Data module type is not specified')
    dm = VifloDataModule.from_specification(spec, action)
    return dm


def make_cropper(crop: str, size: tuple[int, int]):
    if crop == 'random':
        return v2.RandomCrop(size)
    elif crop == 'center':
        return v2.CenterCrop(size)
    else:
        raise ValueError(f'Unknown crop mode: {crop}')


class VifloDataModule(lightning.LightningDataModule):
    name: str

    @classmethod
    def from_specification(cls, spec: Template, action: str) -> typing.Self:
        if cls is not VifloDataModule:
            raise NotImplementedError(f'{cls.from_specification.__qualname__} is not implemented')

        if isinstance(spec.data_module, str):
            dm_type = spec.data_module
        else:
            dm_type = spec.data_module[action]

        subclasses = dnois.utils.subclasses(VifloDataModule)
        for subclass in subclasses:
            if subclass.name == dm_type:
                logger.info(f'Data module type detected: {spec.data_module} (class {subclass.__name__})')
                return subclass.from_specification(spec, action)
        available = [subclass.name for subclass in subclasses]
        raise ValueError(f'Data module type {spec.data_module} is not found, available: {", ".join(available)}')


class BasicDataset(Dataset):
    def __init__(self, root: str | Path, size: tuple[int, int], crop: str = 'center'):
        super().__init__()
        self.root = Path(root)
        self.transform = make_cropper(crop, size)


def make_dataset(config: dict[str, str]):
    config = config.copy()
    cls = None
    cls_name = config.pop('class_name')
    for subclass in dnois.utils.subclasses(BasicDataset):
        if subclass.__name__ == cls_name:
            cls = subclass
    if cls is None:
        raise ValueError(f'Dataset class {cls_name} is not found')
    dataset = cls(**config)
    return dataset


class MixedDataModule(VifloDataModule):
    name = 'mixed'

    def __init__(self, config: dict[str, dict[str, str]], batch_size: int, workers: int):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        train_config = self.config['train']
        random = train_config.pop('random', True)
        dataset = make_dataset(train_config)
        loader = DataLoader(
            dataset, self.batch_size, random, num_workers=self.workers, pin_memory=True, drop_last=True
        )
        return loader

    def val_dataloader(self):
        val_config = self.config['val']
        dataset = make_dataset(val_config)
        loader = DataLoader(
            dataset, self.batch_size, False, num_workers=self.workers, pin_memory=True, drop_last=False
        )
        return loader

    def test_dataloader(self):
        test_config = self.config['test']
        dataset = make_dataset(test_config)
        loader = DataLoader(
            dataset, self.batch_size, False, num_workers=self.workers, pin_memory=True, drop_last=False
        )
        return loader

    @classmethod
    def from_specification(cls, spec: Template, action: str) -> typing.Self:
        dm = cls(
            spec.mixed_data_module_config,
            spec.batch_size,
            spec.workers,
        )
        logger.info(f'Data module ({cls.__name__}) created')
        return dm
