import time

from loguru import logger


def timed_execution(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_seconds = end - start

        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = elapsed_seconds % 60

        logger.info(
            f"{func.__name__}() executed in: {hours} hours, {minutes} minutes, {seconds:.3f} seconds"
        )
        return result

    return wrapper
