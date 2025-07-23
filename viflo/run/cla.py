import argparse
import logging

__all__ = [
    'arg_parser',
    'get_args',
]


def arg_parser():
    parser = argparse.ArgumentParser(
        prog='python -m viflo',
        description='End-to-end design framework of computation imaging system'
    )
    parser.add_argument('spec_file', type=str)
    parser.add_argument(
        '--log-level', type=str, default='INFO', choices=logging.getLevelNamesMapping().keys(),
    )
    subparsers = parser.add_subparsers(dest='action')

    parser_train = subparsers.add_parser('train')
    _add_common_train_arg(parser_train)

    parser_pretrain = subparsers.add_parser('pretrain')
    _add_common_train_arg(parser_pretrain)

    parser_eval = subparsers.add_parser('eval')

    _ = subparsers.add_parser('print')

    return parser


def _add_common_train_arg(parser_train):
    parser_train.add_argument(
        '--no-tensorboard', action='store_false',
        help='Do not start a tensorboard server to monitor the training progress'
    )


def get_args():
    parser = arg_parser()
    args = parser.parse_args()
    return args
