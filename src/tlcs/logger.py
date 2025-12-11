import logging
import typer
from rich.logging import RichHandler

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_suppress=[typer],
                tracebacks_show_locals=False,
            )
        ],
    )

configure_logging()

def get_logger(name):
    return logging.getLogger(name)