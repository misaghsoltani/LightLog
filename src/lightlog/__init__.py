from .decorator import log_prints
from .levelsvalue import CRITICAL, DEBUG, ERROR, INFO, NOTSET, WARNING
from .pylightlog import Logger

__author__ = "Misagh Soltani"
__email__ = "msoltani@email.sc.edu"
__version__ = "0.1.0"
__all__ = ["Logger", "log_prints", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"]
