import logging
import inspect
import logging.config

import yaml
import os

DEFAULT_LOG_LEVEL = logging.DEBUG if __debug__ else logging.INFO

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
