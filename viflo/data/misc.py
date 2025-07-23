from pathlib import Path

from torchvision.io import read_image

from .core import *

__all__ = [
    'SimpleImageDataset',
]


class SimpleImageDataset(BasicDataset):
    pattern = '**/*.png'
    max = 255.

    def __init__(self, root: Path, size: tuple[int, int], crop: str = 'center'):
        super().__init__(root, size, crop)
        self.paths = sorted(self.root.glob(self.pattern))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx].as_posix()
        image = read_image(path)
        image = image.float() / self.max
        image = self.transform(image)  # CHW
        return {'image': image, 'path': path}
