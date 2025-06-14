import dataclasses

import dnois

from e2e.specification import Template


@dataclasses.dataclass
class Design(Template):
    # ========== 一般来说需要改的内容 ==========
    lens_file_path: str = 'structure/lwir.json'
    resolution: tuple[int, int] = (512, 640)
    pixel_size: float = 24e-6
    noise_std: float = 1e-3
    perspective_focal_length: float = 44.605e-3
    wl: tuple[float, ...] = (8e-6, 10e-6, 12e-6)
    depth: float = float('inf')
    image_size: tuple[int, int] = (512, 640)
    batch_size: int = 16
    nn_lr: float = 1e-4
    epochs: int = 50
    warmup_steps: int = 1000
    run_name: str = 'pretrain'

    # ========== 可以忽略的内容 ==========
    optimizable_parameters: tuple[str] = ()
    sampler: int = 256

    segments: tuple[int, int] = (8, 10)
    x_symmetric: bool = True
    y_symmetric: bool = True

    patch_wise_conv_pad: int | tuple[int, int] = 32
    linear_conv: bool = True

    data_root: str = '/public/home/dx/gjq/datasets/HyperSim'
    workers: str = 4

    lr_decay_interval: int = 30

    checkpoint_target: str = 'val/psnr'
    checkpoint_target_mode: str = 'max'
    log_image_interval: int = 2000
