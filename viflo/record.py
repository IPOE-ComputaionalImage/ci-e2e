import contextlib
import logging
import socket
import subprocess

__all__ = [
    'tensorboard_runner',
]

logger = logging.getLogger(__name__)


def is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('localhost', port))
            return True
        except OSError:
            return False


def find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return -1


@contextlib.contextmanager
def tensorboard_runner(logdir: str, port: int, *args):
    # Check if the requested port is available, if not find an available one
    specified = port
    port = find_available_port(port)
    if port != specified:
        if port == -1:
            logger.error('No available port found, tensorboard will not be started')
        else:
            logger.warning(f'Port {specified} is occupied, using port {port} instead')

    cmd = ['tensorboard', '--logdir', logdir, '--port', str(port)]
    cmd = cmd + list(args)
    cmd = ' '.join(cmd)

    logger.info(f'Starting tensorboard server by: {cmd}')
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True,
        text=True,
    )

    try:
        yield  # execute the code in the with block
    finally:
        proc.terminate()
        proc.wait()
        logger.info('Tensorboard server stopped.')
