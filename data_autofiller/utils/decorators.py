from functools import wraps

from line_profiler import LineProfiler

from ..logger import logger


def profile_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        try:
            return profiler(func)(*args, **kwargs)
        finally:
            stats = profiler.get_stats()
            logger.debug(f"Performance profile for {func.__name__}:")
            profiler.print_stats()

    return wrapper
