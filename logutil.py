import logging
import inspect
import logging.config
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import threading
from types import TracebackType

import yaml
import os

DEFAULT_LOG_LEVEL = logging.DEBUG if __debug__ else logging.INFO
_exception_file_handler: RotatingFileHandler | None = None


class DynamicStreamHandler(logging.StreamHandler):
    """Follow pytest's active capture stream without retaining a closed one."""

    def emit(self, record: logging.LogRecord) -> None:
        target = sys.stdout
        if target is None or target.closed:
            target = sys.__stderr__
        if target is None or target.closed:
            return
        self.stream = target
        super().emit(record)


def install_exception_logging(log_path: Path | None = None) -> Path:
    """Persist application errors, including exceptions raised by wx callbacks."""
    global _exception_file_handler
    if log_path is None:
        data_dir = Path(os.environ.get('LOCALAPPDATA') or Path.home()) / 'Pinident'
        log_path = data_dir / 'pinident.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if _exception_file_handler is not None:
        logging.getLogger().removeHandler(_exception_file_handler)
        _exception_file_handler.close()
    handler = RotatingFileHandler(log_path, maxBytes=5_000_000,
                                  backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(name)s:%(funcName)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(handler)
    _exception_file_handler = handler

    def report_exception(exc_type: type[BaseException], exc: BaseException,
                         traceback: TracebackType | None) -> None:
        logging.getLogger('Pinident').critical(
            'Unhandled application exception', exc_info=(exc_type, exc, traceback))

    def report_thread_exception(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not SystemExit:
            logging.getLogger('Pinident').critical(
                'Unhandled exception in thread %s',
                args.thread.name if args.thread is not None else '<unknown>',
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = report_exception
    threading.excepthook = report_thread_exception
    logging.getLogger('Pinident').info('Diagnostic log: %s', log_path)
    return log_path

def open_config_file(filename: str, start_dir: str = None):
    if start_dir is None:
        start_dir = os.getcwd()

    file_path = os.path.join(start_dir, filename)

    if os.path.exists(file_path):
        return open(file_path, 'r')

    parent_dir = os.path.dirname(start_dir)
    if parent_dir == start_dir:
        # Reached the root directory
        raise FileNotFoundError(f"Config file not found: {filename} in any parent directory")

    return open_config_file(filename, parent_dir)

def load_config() -> None:
    logConfigPath = 'logging.yml'

    with open_config_file(logConfigPath) as logConfig:
        config = yaml.safe_load(logConfig.read())
        logConfig.close()

    debugConfigPath = 'logging.debug.yml'
    if debugConfigPath is not None:
        with open_config_file(debugConfigPath) as debugLogConfig:
            debugConfig = yaml.safe_load(debugLogConfig.read())
            config = {**config, **debugConfig}
            debugLogConfig.close()

    logging.config.dictConfig(config)


def get_file_name(file_path: str) -> str | None:
    """Extract the file name from a file path."""
    if not file_path:

        return None
    return file_path.split('/')[-1].split('\\')[-1]  # Handle both Unix and Windows paths

def getLog() -> logging.Logger:
    # Get the caller's frame (1 level up)
    frame = inspect.currentframe().f_back
    ct = frame.f_code
    parts = ct.co_qualname.split('.')
    method_name = parts[0]
    if len(parts) == 1:
        # If no module or class name, use the function name directly
        method_name = get_file_name(ct.co_filename)
    # if len(parts) > 1:
    #     method_name = '.'.join(parts[-2:])
    # else:
    #     # Remove redundant prefix if present (e.g., "func:func")
    #     method_name = ct.co_qualname.split(':')[-1]
    logger = logging.getLogger(method_name)
    logger.setLevel(DEFAULT_LOG_LEVEL)
    return logger

load_config()
