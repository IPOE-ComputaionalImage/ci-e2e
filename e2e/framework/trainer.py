import logging

import lightning
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

import e2e.conf

from e2e.specification import Template

__all__ = [
    'create_trainer_from_spec',
]

logger = logging.getLogger(__name__)


def create_trainer_from_spec(spec: Template) -> lightning.Trainer:
    tb = TensorBoardLogger(e2e.conf.optimization_result_dir, spec.run_name, default_hp_metric=False)
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
