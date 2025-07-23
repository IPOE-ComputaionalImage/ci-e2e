import logging

import lightning
import torch

from .cla import get_args
from ..specification import parse_spec_file
from ..viflog import setup_log

__all__ = [
    'main',
]


logger = logging.getLogger(__name__)


def setup_global(spec):
    lightning.seed_everything(spec.random_seed)
    torch.set_float32_matmul_precision('high')
    spec.setup_global()


def main():
    args = get_args()

    setup_log(args.log_level)

    spec = parse_spec_file(args.spec_file)
    logger.info('Design file resolved')

    setup_global(spec)

    framework = spec.create_framework()
    logger.info(f'Prepare to execute action: {args.action}')

    kwargs = {}
    if args.action == 'train':
        kwargs['tensorboard_on'] = not args.no_tensorboard
    framework.execute(args.action, spec, **kwargs)
    logger.info('Action completed')

    logger.info('Done')
