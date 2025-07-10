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

    parser_eval = subparsers.add_parser('eval')

    return parser


def get_args():
    parser = arg_parser()
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    log_level = logging.getLevelName(args.log_level.upper())
    logging.basicConfig(
        format='[%(asctime)s](%(levelname)s)%(threadName)s/%(name)s:%(message)s',
        level=log_level
    )

    spec = parse_spec_file(args.spec_file)
    logger.info('Design file resolved')

    lightning.seed_everything(spec.random_seed)
    torch.set_float32_matmul_precision('high')

    framework = CIFramework.from_specification(spec)
    logger.info(f'Prepare to execute action: {args.action}')
    framework.execute(args.action, spec)
    logger.info('Action completed')

    logger.info('Done')


if __name__ == '__main__':
    main()
