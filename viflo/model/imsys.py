import logging
import typing

import dnois
import torch

import viflo.nn
from viflo.specification import Template, FromSpecification

__all__ = [
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


class ImagingSystem(torch.nn.Module, FromSpecification):
    nn_enabled = True

    def __init__(
        self,
        cam: dnois.Camera,
        conv_pad: int = 32,
        linear_conv: bool = True,
        imaging_in_linear: bool = True,
        random_fov: bool = False,
    ):
        super().__init__()
        self.c = cam
        self.nn = viflo.nn.WienerUNet(self.nn_enabled)

        self.conv_pad = conv_pad
        self.linear_conv = linear_conv
        self.imaging_in_linear = imaging_in_linear
        self.random_fov = random_fov

    def forward(self, image, **kwargs):
        if self.imaging_in_linear:
            image = dnois.isp.srgb2linear(image)  # 转换到线性空间
        scene = dnois.scene.ImageScene(image)  # 包装为场景对象

        segments = kwargs.get('segments', self.o.segments)
        if isinstance(segments, tuple):
            kwargs.setdefault('pad', self.conv_pad)  # 分块卷积时每块的padding大小（32即块与块重叠64）
            kwargs.setdefault('linear_conv', self.linear_conv)  # 分块卷积时是否使用线性卷积
        elif segments == 'uniform':
            if self.random_fov:
                kwargs.setdefault('fov', 'random')

        captured = self.c(scene, optics_kw=kwargs)  # 调用Camera对象时通过optics_kw参数传入光学系统的参数

        result = self.nn(captured, self.o)
        pred = result['pred']

        if self.imaging_in_linear:
            pred = dnois.isp.linear2srgb(pred)
            captured = dnois.isp.linear2srgb(captured)
        return pred, captured, result

    @property
    def o(self) -> dnois.optics.rt.CoaxialRayTracing:
        return self.c.optics  # noqa

    @classmethod
    def from_specification(cls, spec: Template) -> typing.Self:
        sq = create_sq(spec)
        sensor = spec.create_sensor()
        optics = spec.create_optics(sq, sensor)

        obj = cls(
            dnois.Camera(optics, sensor),
            spec.patch_wise_conv_pad,
            spec.linear_conv,
            spec.imaging_in_linear,
            spec.random_fov,
        )
        logger.info('Imaging system model has been created')
        return obj
