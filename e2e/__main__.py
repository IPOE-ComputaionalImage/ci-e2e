import argparse
import logging

import lightning
import torch

from e2e.framework import CIFramework
from e2e.specification import parse_spec_file

logger = logging.getLogger(__name__)


def arg_parser():
    parser = argparse.ArgumentParser(
        prog='python -m e2e',
        description='End-to-end design framework of computation imaging system'
    )
    parser.add_argument('spec_file', type=str)
    parser.add_argument(
        '--log-level', type=str, default='INFO', choices=logging.getLevelNamesMapping().keys(),
    )
    subparsers = parser.add_subparsers(dest='action')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument(
        '--no-tensorboard', action='store_false',
        help='Do not start a tensorboard server to monitor the training progress'
    )

    parser_eval = subparsers.add_parser('eval')

    return parser


def get_args():
    parser = arg_parser()
    args = parser.parse_args()
    return args


def setup_log(level: str):
    log_level = logging.getLevelName(level)
    logging.basicConfig(
        format='[%(asctime)s](%(levelname)s)%(threadName)s/%(name)s:%(message)s',
        level=log_level
    )


def setup_global(spec):
    lightning.seed_everything(spec.random_seed)
    torch.set_float32_matmul_precision('high')

    spec.set_unit()
    spec.load_catalogs()
    spec.setup_physical_environment()


def main():
    args = get_args()

    setup_log(args.log_level.upper())

    spec = parse_spec_file(args.spec_file)
    logger.info('Design file resolved')

    setup_global(spec)

    framework = CIFramework.from_specification(spec)
    logger.info(f'Prepare to execute action: {args.action}')

    kwargs = {}
    if args.action == 'train':
        kwargs['tensorboard_on'] = not args.no_tensorboard
    framework.execute(args.action, spec, **kwargs)
    logger.info('Action completed')

    logger.info('Done')


if __name__ == '__main__':
    main()
