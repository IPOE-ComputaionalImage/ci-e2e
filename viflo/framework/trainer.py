import contextlib
import logging

import lightning
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

import viflo.conf
from viflo.record import tensorboard_runner
from viflo.specification import Template

__all__ = [
    'create_trainer_from_spec',
    'train_context_manager',
]

logger = logging.getLogger(__name__)


def create_trainer_from_spec(spec: Template) -> lightning.Trainer:
    tb = TensorBoardLogger(viflo.conf.optimization_result_dir, spec.run_name, default_hp_metric=False)
    trainer = lightning.Trainer(
        logger=tb,
        callbacks=[
            lightning.pytorch.callbacks.ModelCheckpoint(
                monitor=spec.checkpoint_target, mode=spec.checkpoint_target_mode, save_last=True
            ),
            lightning.pytorch.callbacks.LearningRateMonitor(),
            lightning.pytorch.callbacks.OnExceptionCheckpoint(tb.log_dir),
        ],
        max_epochs=spec.epochs,
        enable_progress_bar=spec.enable_progress_bar,
        deterministic=True,
        profiler=SimpleProfiler(tb.log_dir, 'profiler')
    )

    logger.info('Trainer created')
    return trainer


def train_context_manager(spec: Template, trainer: lightning.Trainer, tensorboard_on: bool = True):
    if not tensorboard_on:
        return contextlib.nullcontext()

    tb_loggers = [_logger for _logger in trainer.loggers if isinstance(_logger, TensorBoardLogger)]
    if len(tb_loggers) == 0:
        logger.warning('No tensorboard logger is set, tensorboard will not be started')
        return contextlib.nullcontext()
    tb_logger = tb_loggers[0]

    if spec.tensorboard_logdir_level == 'top':
        log_dir = tb_logger.save_dir
    elif spec.tensorboard_logdir_level == 'run':
        log_dir = tb_logger.root_dir
    elif spec.tensorboard_logdir_level == 'version':
        log_dir = tb_logger.log_dir
    else:
        logger.warning(f'Unknown tensorboard_logdir_level {spec.tensorboard_logdir_level}, fall back to "version"')
        log_dir = tb_logger.log_dir

    cla = spec.tensorboard_cla(log_dir)  # command line arguments
    return tensorboard_runner(cla)
