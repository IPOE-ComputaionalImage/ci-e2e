import csv
import logging
import typing
import warnings
from pathlib import Path
from typing import Literal

import lightning
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import e2e.specification
from e2e.specification import Template

__all__ = [
    'HyperSim',
    'HyperSimDM',
]

logger = logging.getLogger(__name__)


class HyperSim(Dataset):
    height = 768
    height_min = height
    height_max = height
    width = 1024
    width_min = width
    width_max = width
    size = {'train': 59543, 'val': 7386, 'test': 7690}
    float_focal = 886.81
    depth_unit = 'm'
    max_depth = 1820.035888671875

    def __init__(
        self,
        root: str | Path,
        split: Literal['train', 'val', 'test'],
        size: tuple[int, int],
        split_file_path: str | Path = None,
        use_hdr: bool = False,
        use_tonemap: bool = True,
        nan_fill_value: float = None,
        crop: str = 'random',
    ):
        super().__init__()
        root = Path(root)
        if split_file_path is None:
            split_file_path = root / 'meta' / 'metadata_images_split_scene_v1.csv'
        else:
            split_file_path = Path(split_file_path)

        self._root = root
        self.use_hdr = use_hdr
        self.use_tonemap = use_tonemap
        self.nan_fill_value = nan_fill_value
        with open(split_file_path, 'r') as f:
            reader = csv.reader(f)
            foi = [line for line in reader if line[5] == split]  # files of interest
        self._foi = [(line[0], line[1], int(line[2])) for line in foi]  # content: (scene_name, camera_name, frame_id)
        if len(self._foi) != self.size[split]:
            warnings.warn(f'Wrong number of samples for {split} split '
                          f'of HyperSim: {len(self._foi)} ({self.size[split]} expected)')

        if crop == 'random':
            self.transform = v2.RandomCrop(size)
        elif crop == 'center':
            self.transform = v2.CenterCrop(size)
        else:
            raise ValueError(f'Unknown crop mode: {crop}')

    def __len__(self) -> int:
        return len(self._foi)

    def __getitem__(self, item: int):
        path_info = self._foi[item]
        frame_id = f'{path_info[2]:04d}'
        path = self._root / path_info[0] / 'images'

        if self.use_hdr:
            i_path_cam = f'scene_{path_info[1]}_final_hdf5'
            i_path_prefix = 'hdf5'
            raise NotImplementedError()
        else:
            i_path_cam = f'scene_{path_info[1]}_final_preview'
            i_path_prefix = 'jpg'
        if self.use_tonemap:
            i_path_type = 'tonemap'
        else:
            i_path_type = 'color'
        i_path = path / i_path_cam / f'frame.{frame_id}.{i_path_type}.{i_path_prefix}'

        if i_path.suffix == '.hdf5':
            raise NotImplementedError()
        else:
            image = np.array(Image.open(i_path))
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            image = image / 255.

        image = self.transform(image)
        return {
            'image': image,
            'idx': item,
            'path': i_path.as_posix(),
            # 'ipath':i_path,
        }


class HyperSimDM(lightning.LightningDataModule, e2e.specification.FromSpecification):
    def __init__(self, root: str | Path, image_size: tuple[int, int], batch_size: int, workers: int):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.workers = workers

    def dataset(self, split):
        return HyperSim(self.root, split, self.image_size)

    def train_dataloader(self):
        return DataLoader(
            self.dataset('train'), self.batch_size, True, num_workers=self.workers, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset('val'), self.batch_size, False, num_workers=self.workers, pin_memory=True,
        )

    @classmethod
    def from_specification(cls, spec: Template) -> typing.Self:
        dm = cls(
            spec.data_root,
            spec.image_size,
            spec.batch_size,
            spec.workers,
        )
        logger.info('Data module created')
        return dm
