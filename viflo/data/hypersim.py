import csv
import logging
import typing
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

from .core import *
from ..specification import Template, FromSpecification

__all__ = [
    'HyperSim',
    'HyperSimDM',
]

logger = logging.getLogger(__name__)


def _filter_unavailable(foi_list: list[tuple[str, str, int]], root):
    def is_available(path_info: tuple[str, str, int]):
        path = root / path_info[0] / 'images'
        frame_id = f'{path_info[2]:04d}'
        i_path_cam = f'scene_{path_info[1]}_final_preview'
        i_path_prefix = 'jpg'
        i_path = path / i_path_cam / f'frame.{frame_id}.color.{i_path_prefix}'
        return i_path.exists()

    foi_list = filter(is_available, foi_list)
    foi_list = list(foi_list)
    return foi_list


class HyperSim(BasicDataset):
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
        super().__init__(root, size, crop)
        if split_file_path is None:
            split_file_path = self.root / 'meta' / 'metadata_images_split_scene_v1.csv'
        else:
            split_file_path = Path(split_file_path)

        self.use_hdr = use_hdr
        self.use_tonemap = use_tonemap
        self.nan_fill_value = nan_fill_value
        with open(split_file_path, 'r') as f:
            reader = csv.reader(f)
            foi = [line for line in reader if line[5] == split]  # files of interest
        foi_list = [(line[0], line[1], int(line[2])) for line in foi]  # content: (scene_name, camera_name, frame_id)
        self._foi = _filter_unavailable(foi_list, self.root)
        if len(self._foi) != self.size[split]:
            logger.warning(f'Wrong number of samples for {split} split '
                           f'of HyperSim: {len(self._foi)} ({self.size[split]} expected)')

    def __len__(self) -> int:
        return len(self._foi)

    def __getitem__(self, item: int):
        path_info = self._foi[item]
        frame_id = f'{path_info[2]:04d}'
        path = self.root / path_info[0] / 'images'

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


class HyperSimDM(VifloDataModule, FromSpecification):
    name = 'hypersim'

    def __init__(self, root: str | Path, image_size: tuple[int, int], batch_size: int, workers: int):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.workers = workers

    def dataset(self, split, random_crop: bool = True):
        return HyperSim(self.root, split, self.image_size, crop='random' if random_crop else 'center')

    def train_dataloader(self):
        return make_loader(self.dataset('train'), self.batch_size, True, self.workers)

    def val_dataloader(self):
        return make_loader(self.dataset('val', False), self.batch_size, False, self.workers)

    def test_dataloader(self):
        return make_loader(self.dataset('test', False), self.batch_size, False, self.workers)

    @classmethod
    def from_specification(cls, spec: Template, _) -> typing.Self:
        dm = cls(
            spec.data_root,
            spec.hypersim_image_size,
            spec.batch_size,
            spec.workers,
        )
        logger.info('Data module created')
        return dm
