import logging
from logging.config import dictConfig
from .config import settings

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        },
    },
    "root": {
        "handlers": ["console"],
        "level": settings.log_level
    },
}

def init_logging():
    dictConfig(LOGGING)
