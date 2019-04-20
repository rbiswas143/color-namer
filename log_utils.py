import logging.handlers
import os
import sys

import config

# Create log directory if it does not exist
os.makedirs(config.LOG_DIR, exist_ok=True)

# File Handler

_file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] || %(module)s :: %(funcName)s :: %(lineno)s || %(message)s"
)
file_handler = logging.FileHandler(os.path.join(config.LOG_DIR, config.LOG_FILE_NAME))
file_handler.setFormatter(_file_formatter)
file_handler.setLevel(config.LOG_LEVEL_FILE)

# CLI Handler

_cli_formatter = logging.Formatter("%(message)s")
cli_handler = logging.StreamHandler(stream=sys.stdout)
cli_handler.setFormatter(_cli_formatter)
cli_handler.setLevel(config.LOG_LEVEL_CLI)

# Logger

log = logging.getLogger('color')
log.addHandler(file_handler)
log.addHandler(cli_handler)
log.setLevel('DEBUG')
