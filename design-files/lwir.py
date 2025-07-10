# ========== 一般来说需要改的内容 ==========
lens_file_path = 'structure/lwir.json'  # 透镜结构文件路径
optimizable_parameters = [  # 可优化的参数
    '2.curvature',  # 2表示第2面（最前面的为0面），curvature表示曲率
    '2.a2',  # 非球面系数4次项
    '2.a3',  # 6次项
    '2.a4',
    '3.a2',
    '3.a3',
    '3.a4',
    '4.context.distance',  # 第4面到下一面（这里是像面）的距离
]
# 参数变换
parameter_transformations = {
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
}
resolution = (512, 640)  # 传感器分辨率
pixel_size = 24e-6  # 像素尺寸
noise_std = 1e-3  # 高斯噪声标准差
perspective_focal_length = 44.605e-3
wl = (8e-6, 10e-6, 12e-6)
depth = float('inf')
image_size = (512, 640)
batch_size = 2
nn_lr = 1e-4
epochs = 5
run_name = 'lwir'
nn_init_path = 'pretrained.pth'
data_root = r'\\wsl.localhost\Ubuntu\home\hcc\datasets\HyperSim'
trained_ckpt_path = 'tapes/lwir/version_0/checkpoints/last.ckpt'

target_focal_length = perspective_focal_length
focal_length_loss_weight = 1e6

# ========== 可以忽略的内容 ==========
framework = 'e2e'

sampler = 256

segments = (8, 10)
x_symmetric = True
y_symmetric = True

patch_wise_conv_pad = 32
linear_conv = True

workers = 4

lr_decay_interval = 30

checkpoint_target = 'val/psnr'
checkpoint_target_mode = 'max'
log_image_interval = 2000
