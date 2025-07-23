import logging

from colorama import Fore, Style, just_fix_windows_console

__all__ = [
    'color_fmt',
    'setup_log',

    'ColoredFormatter',
]


def color_fmt(msg, *flags):
    return ''.join(flags) + msg + Style.RESET_ALL


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        formatted = super().format(record)

        if record.levelno == logging.INFO:
            return color_fmt(formatted, Fore.CYAN)
        elif record.levelno == logging.WARNING:
            return color_fmt(formatted, Fore.YELLOW)
        elif record.levelno == logging.ERROR:
            return color_fmt(formatted, Fore.RED)
        elif record.levelno == logging.CRITICAL:
            return color_fmt(formatted, Fore.RED, Style.BRIGHT)
        elif record.levelno == logging.DEBUG:
            return color_fmt(formatted, Fore.GREEN)

        return formatted


def setup_log(level: str):
    just_fix_windows_console()

    root_logger = logging.getLogger()
    log_level = getattr(logging, level.upper())
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    formatter = ColoredFormatter(
        '[%(asctime)s](%(levelname)s)%(threadName)s/%(name)s: %(message)s'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
