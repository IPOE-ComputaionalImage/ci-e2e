import logging
import typing

import dnois
from dnois.optics import rt
import torch

import e2e.nn
from e2e.specification import Template, FromSpecification

__all__ = [
    'create_sensor',
    'create_sq',

    'ImagingSystem',
]

logger = logging.getLogger(__name__)


def create_sq(spec: Template):
    sq = spec.create_lens()

    spec.setup_optimizable_parameters(sq)
    if spec.log_optimizable_parameters:
        optimizable_names = [k for k, v in sq.named_parameters() if v.requires_grad]
        logger.info(f'Optimizable parameters in lens: {optimizable_names}')

    spec.setup_parameter_transformations(sq)
    return sq


def create_sensor(spec: Template) -> dnois.sensor.StandardSensor:
    sensor = dnois.sensor.StandardSensor(
        spec.resolution,
        spec.pixel_size,
        spec.rgb_sensor,
        None,
        spec.bayer_pattern,
        spec.noise_std,
        spec.max_value,
        spec.quantize,
        spec.linear2srgb,
    )
    return sensor


class ImagingSystem(torch.nn.Module, FromSpecification):
    nn_enabled = True

    def __init__(
        self,
        cam: dnois.Camera,
        conv_pad: int = 32,
        linear_conv: bool = True,
        imaging_in_linear: bool = True
    ):
        super().__init__()
        self.c = cam
        self.nn = e2e.nn.WienerUNet(self.nn_enabled)

        self.conv_pad = conv_pad
        self.linear_conv = linear_conv
        self.imaging_in_linear = imaging_in_linear

    def forward(self, image, **kwargs):
        if self.imaging_in_linear:
            image = dnois.isp.srgb2linear(image)  # 转换到线性空间
        scene = dnois.scene.ImageScene(image)  # 包装为场景对象

        if isinstance(kwargs.get('segments', self.o.segments), tuple):
            kwargs.setdefault('pad', self.conv_pad)  # 分块卷积时每块的padding大小（32即块与块重叠64）
            kwargs.setdefault('linear_conv', self.linear_conv)  # 分块卷积时是否使用线性卷积
        captured = self.c(scene, optics_kw=kwargs)  # 调用Camera对象时通过optics_kw参数传入光学系统的参数

        axial_inf = self.o.new_tensor([0, 0, float('inf')])
        psf4deconv = self.o.psf(axial_inf).unsqueeze(0)  # 计算轴上无穷远的PSF
        pred = self.nn(captured, psf4deconv)

        if self.imaging_in_linear:
            pred = dnois.isp.linear2srgb(pred)
            captured = dnois.isp.linear2srgb(captured)
        return pred, captured

    @property
    def o(self) -> dnois.optics.rt.CoaxialRayTracing:
        return self.c.optics  # noqa

    @classmethod
    def from_specification(cls, spec: Template) -> typing.Self:
        sq = create_sq(spec)

        sensor = create_sensor(spec)

        sampling_num = dnois.typing.size2d(spec.sampler)
        sampler = sq.first.apt.sampler('rect', sampling_num)
        psf_center = rt.RobustMeanPsfCenter()
        psf_model = rt.IncoherentRectKernelPsf(psf_center=psf_center)
        optics = rt.CoaxialRayTracing(
            sq,
            sensor,
            perspective_focal_length=spec.perspective_focal_length,
            psf_model=psf_model,
            sampler=sampler,
            wl=spec.wl,
            segments=spec.segments,
            depth=spec.depth,
            psf_size=spec.psf_size,
            norm_psf=spec.norm_psf,
            cropping=spec.cropping,
            x_symmetric=spec.x_symmetric,
            y_symmetric=spec.y_symmetric,
        )
        if spec.log_optics_info:
            logger.info(f'Optics created: {optics}')

        obj = cls(
            dnois.Camera(optics, sensor),
            spec.patch_wise_conv_pad,
            spec.linear_conv,
            spec.imaging_in_linear,
        )
        logger.info('Imaging system model has been created')
        return obj
