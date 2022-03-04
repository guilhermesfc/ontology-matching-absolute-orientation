LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - [%(levelname)s] %(name)s: %(message)s"}
    },
    "handlers": {
        "default": {
            "level": "DEBUG",  # -> overrules definition in loggers !
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": "logging.log",
            "maxBytes": 1024,
            "backupCount": 3,
        },
    },
    "loggers": {
        "": {  # root logger -> overrules all following
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True,
        }
    },
}
