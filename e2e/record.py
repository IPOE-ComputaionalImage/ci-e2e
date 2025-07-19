import contextlib
import logging

import subprocess

__all__ = [
    'tensorboard_runner',
]

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def tensorboard_runner(args: str):
    cmd = 'tensorboard ' + args
    logger.info(f'Starting tensorboard server by: {cmd}')
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True,
        text=True,
    )

    yield  # execute the code in the with block

    proc.terminate()
    proc.wait()
    logger.info('Tensorboard server stopped.')
