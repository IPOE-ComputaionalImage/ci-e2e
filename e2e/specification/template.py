import dataclasses
import logging
import typing

import dnois
from dnois.optics import rt
import torch

import e2e.material

__all__ = [
    'Template',
]

logger = logging.getLogger(__name__)


def _get_submodule_surf(qualified_param_name: str, module: torch.nn.Module) -> dnois.torch.EnhancedModule:
    qualified_param_name = '.'.join(qualified_param_name.split('.')[:-1])
    return typing.cast(dnois.torch.EnhancedModule, module.get_submodule(qualified_param_name))


def _scale_param(m: dnois.torch.EnhancedModule, param_name: str, log: bool):
    value = m.get_parameter(param_name)
    if not value.ndim == 0:
        logger.warning(f'Parameter {param_name} is not a scalar and hence cannot be scaled')
        return

    value = value.item()
    if value == 0:
        logger.warning(f'Parameter {param_name} is zero and hence cannot be scaled')
        return

    t = dnois.torch.Transform.scale(value)
    m.set_transform(param_name, t)
    if log:
        logger.info(f'Set transformation for parameter {param_name}: {t.__class__.__name__}')


# TODO: provide more methods to override
@dataclasses.dataclass
class Template:
    # ---------- framework ----------
    framework: str = 'e2e'

    # ---------- create surface sequence ----------
    lens_file_path: str = None

    def create_lens(self) -> rt.CoaxialSurfaceSequence:
        if self.lens_file_path is None:
            raise ValueError('lens_file_path is not specified')

        logger.info(f'Loading lens from {self.lens_file_path}')
        while True:
            try:
                return rt.CoaxialSurfaceSequence.load_json(self.lens_file_path)
            except dnois.mt.MaterialNotFoundError as e:
                name = e.material_name
                catalog = e2e.material.find_material(name)
                e2e.material.load_catalog(catalog)
                logger.info(f'Loaded catalog {catalog} to use material {name}')

    # ---------- specify parameters to optimize ----------
    # NOTE: only in optical system model
    optimizable_parameters: list[str] = None
    frozen_parameters: list[str] = None
    log_optimizability_modified: bool = True
    log_optimizable_parameters: bool = True

    def setup_optimizable_parameters(self, sq: rt.CoaxialSurfaceSequence):
        freeze = self.frozen_parameters is not None
        unfreeze = self.optimizable_parameters is not None
        if freeze and unfreeze:
            raise ValueError('optimizable_parameters and frozen_parameters cannot be specified at the same time')
        elif not freeze and not unfreeze:
            logger.info('Optimize all parameters by default')

        if freeze:
            if self.log_optimizability_modified:
                logger.info(f'Freezing parameters: {self.frozen_parameters}')
            sq.unfreeze()
            for param_name in self.frozen_parameters:
                sq.freeze(param_name)

        if unfreeze:
            if self.log_optimizability_modified:
                logger.info(f'Enable optimization of parameters: {self.optimizable_parameters}')
            sq.freeze()
            for param_name in self.optimizable_parameters:
                sq.unfreeze(param_name)

    # ---------- set up parameter transformations ----------
    parameter_transformations: dict[str, dnois.torch.Transform | str] = None
    log_param_transformations: bool = True

    def setup_parameter_transformations(self, sq: rt.CoaxialSurfaceSequence):
        if self.parameter_transformations is None or len(self.parameter_transformations) == 0:
            logger.info('No parameter transformations specified')
            return

        for k, v in self.parameter_transformations.items():
            m = _get_submodule_surf(k, sq)
            param_name = k.split('.')[-1]
            if isinstance(v, str):
                if v == 'scale':
                    _scale_param(m, param_name, self.log_param_transformations)
                else:
                    raise ValueError(f'Unknown parameter transformation: {v}')
            else:
                m.set_transform(param_name, v)
                if self.log_param_transformations:
                    logger.info(f'Set transformation for parameter {k}: {v.__class__.__name__}')

    # ---------- materials ----------
    required_catalogs: list[str] = None

    def load_catalogs(self):
        if self.required_catalogs is None:
            return
        for catalog in self.required_catalogs:
            e2e.material.load_catalog(catalog)

    # ---------- sensor parameters ----------
    resolution: int | tuple[int, int] = None
    pixel_size: float | tuple[float, float] = None
    rgb_sensor: bool = True
    bayer_pattern: dnois.sensor.BayerPattern = None
    noise_std: float | tuple[float, float] = None
    max_value: float = 1.
    quantize: int = 0
    linear2srgb: bool = False

    def create_sensor(self) -> dnois.sensor.StandardSensor:
        sensor = dnois.sensor.StandardSensor(
            self.resolution,
            self.pixel_size,
            self.rgb_sensor,
            None,
            self.bayer_pattern,
            self.noise_std,
            self.max_value,
            self.quantize,
            self.linear2srgb,
        )
        return sensor

    # ---------- configuration of optical system ----------
    perspective_focal_length: float = None
    psf_center: rt.PsfCenter = 'mean-robust'
    sampler: int | tuple[int, int] = None
    wl: dnois.typing.Vector = dnois.fdc()
    segments: dnois.optics.SegLit | int | tuple[int, int] = 'uniform'
    random_fov: bool = False
    depth: dnois.typing.Vector = float('inf')
    psf_size: int | tuple[int, int] = 64
    norm_psf: bool = True
    cropping: int | tuple[int, int] = 0
    x_symmetric: bool = False
    y_symmetric: bool = False
    log_optics_info: bool = True

    def create_optics(self, sq, sensor):
        sampling_num = dnois.typing.size2d(self.sampler)
        sampler = sq.first.apt.sampler('rect', sampling_num)
        psf_center = rt.RobustMeanPsfCenter()
        psf_model = rt.IncoherentRectKernelPsf(psf_center=psf_center)
        optics = rt.CoaxialRayTracing(
            sq,
            sensor,
            perspective_focal_length=self.perspective_focal_length,
            psf_model=psf_model,
            sampler=sampler,
            wl=self.wl,
            segments=self.segments,
            depth=self.depth,
            psf_size=self.psf_size,
            norm_psf=self.norm_psf,
            cropping=self.cropping,
            x_symmetric=self.x_symmetric,
            y_symmetric=self.y_symmetric,
        )

        if self.log_optics_info:
            logger.info(f'Optics created: {optics}')
        return optics

    # ---------- configuration of image formation ----------
    patch_wise_conv_pad: int | tuple[int, int] = 32
    linear_conv: bool = True
    imaging_in_linear: bool = True

    # ---------- dataset ----------
    data_root: str = None
    image_size: int | tuple[int, int] = None
    batch_size: int = None
    workers: int = None

    # ---------- configuration of optimization ----------
    optics_lr: float = 1e-3
    nn_lr: float = 5e-5
    lr_decay_factor: float = 0.1
    lr_decay_interval: int = 50
    epochs: int = None
    warmup_steps: int = 0

    # ---------- recording ----------
    run_name: str = None
    checkpoint_target: str = None
    checkpoint_target_mode: str = None
    enable_progress_bar: bool = True
    log_image_interval: int = None

    # ---------- tensorboard ----------
    tensorboard_logdir_level: typing.Literal['top', 'run', 'version'] = 'version'
    tensorboard_port: int = None

    def tensorboard_cla(self, log_dir: str) -> str:
        return f'--logdir {log_dir} --port {self.tensorboard_port}'

    # ---------- initialization ----------
    nn_init_path: str = None

    # ---------- constraints ----------
    target_focal_length: float = None
    focal_length_loss_weight: float = None

    # ---------- evaluation ----------
    trained_ckpt_path: str = None

    # ---------- physical environment ----------
    system_temperature: float = None
    system_pressure: float = None
    temperature_affect_n: bool = False
    pressure_affect_n: bool = False

    def setup_physical_environment(self):
        dnois.conf.temperature_affect_n = self.temperature_affect_n
        dnois.conf.pressure_affect_n = self.pressure_affect_n
        if self.system_temperature is not None:
            dnois.conf.default_temperature = self.system_temperature
        if self.system_pressure is not None:
            dnois.conf.default_pressure = self.system_pressure

    # ---------- unit ----------
    length_unit: str = 'm'

    def set_unit(self):
        dnois.set_default('length', self.length_unit)

    # ---------- miscellaneous ----------
    random_seed: int = 42
