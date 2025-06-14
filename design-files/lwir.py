import dataclasses

from e2e.specification import Template


@dataclasses.dataclass
class Design(Template):
    # ========== 一般来说需要改的内容 ==========
    lens_file_path: str = 'structure/lwir.json'  # 透镜结构文件路径
    optimizable_parameters: tuple[str] = (  # 可优化的参数
        '2.curvature',  # 2表示第2面（最前面的为0面），curvature表示曲率
        '2.a2',  # 非球面系数4次项
        '2.a3',  # 6次项
        '2.a4',
        '3.a2',
        '3.a3',
        '3.a4',
        '4.context.distance',  # 第4面到下一面（这里是像面）的距离
    )
    # 参数变换
    parameter_transformations: dict = dataclasses.field(default_factory=lambda: {
        # scale表示缩放，即参数初始值为x_0时，
        # 令实际优化变量的初值为1，计算时将其乘以x_0得到表观值
        '2.curvature': 'scale',
        '2.a2': 'scale',
        '2.a3': 'scale',
        '2.a4': 'scale',
        '3.a2': 'scale',
        '3.a3': 'scale',
        '3.a4': 'scale',
        '4.context.distance': 'scale',
    })
    resolution: tuple[int, int] = (512, 640)  # 传感器分辨率
    pixel_size: float = 24e-6  # 像素尺寸
    noise_std: float = 1e-3  # 高斯噪声标准差
    perspective_focal_length: float = 44.605e-3
    wl: tuple[float, ...] = (8e-6, 10e-6, 12e-6)
    depth: float = float('inf')
    image_size: tuple[int, int] = (512, 640)
    batch_size: int = 16
    nn_lr: float = 1e-4
    epochs: int = 50
    run_name: str = 'lwir'
    nn_init_path: str = 'pretrained.pth'
    target_focal_length: float = perspective_focal_length
    focal_length_loss_weight: float = 1e6

    # ========== 可以忽略的内容 ==========
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
