import logging
import inspect
import logging.config

import yaml

DEFAULT_LOG_LEVEL = logging.DEBUG if __debug__ else logging.INFO

def load_config() -> None:
    logConfigPath = 'logging.yml'
    with open(logConfigPath, 'r') as logConfig:
        config = yaml.safe_load(logConfig.read())
        logConfig.close()

    debugConfigPath = 'logging.debug.yml'
    if debugConfigPath is not None:
        with open(debugConfigPath, 'r') as debugLogConfig:
            debugConfig = yaml.safe_load(debugLogConfig.read())
            config = {**config, **debugConfig}
            debugLogConfig.close()

    logging.config.dictConfig(config)


def getLog() -> logging.Logger:
    # Get the caller's frame (1 level up)
    frame = inspect.currentframe().f_back
    method_name = frame.f_code.co_name
    logger = logging.getLogger(method_name)
    logger.setLevel(DEFAULT_LOG_LEVEL)
    return logger

load_config()
