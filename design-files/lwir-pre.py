# ========== 一般来说需要改的内容 ==========
lens_file_path = 'structure/lwir.json'
resolution = (512, 640)
pixel_size = 24e-6
noise_std = 1e-3
perspective_focal_length = 44.605e-3
wl = (8e-6, 10e-6, 12e-6)
depth = float('inf')
image_size = (512, 640)
batch_size = 2
nn_lr = 1e-4
epochs = 50
warmup_steps = 1000
run_name = 'pretrain'
data_root = r'\\wsl.localhost\Ubuntu\home\hcc\datasets\HyperSim'
# trained_ckpt_path = 'pretrained.pth'

# ========== 可以忽略的内容 ==========
framework = 'pretrain'

optimizable_parameters = []
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
