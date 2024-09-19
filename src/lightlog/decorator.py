import functools
import inspect
from typing import Any, Callable, Optional

from .pylightlog import Logger


def log_prints(name: Optional[str] = None,
               file_path: Optional[str] = None,
               mode: str = 'a',
               level: int = -1,
               use_rank: bool = False,
               rank: Optional[int] = None,
               world_size: Optional[int] = None,
               auto_detect_env: Optional[str] = None,
               logger_instance: Optional['Logger'] = None):
    """
    A decorator that redirects all print statements in the decorated function or class to a logger.

    This decorator can be used with functions, classes, and class methods. It creates a new Logger
    instance or uses an existing one to capture all print statements made within the decorated
    object.

    Args:
        name (Optional[str]): The name of the logger instance.
        file_path (Optional[str]): Path to the log file.
        mode (str): File mode for writing logs, either 'a' (append) or 'w' (overwrite). Default
            is 'a'.
        level (int): The log level. Default is -1 (no level filtering).
        use_rank (bool): If True, includes process rank in the logs. Default is False.
        rank (Optional[int]): The rank of the process in a distributed system.
        world_size (Optional[int]): The total number of processes in a distributed system.
        auto_detect_env (Optional[str]): Specifies how to automatically detect the environment for
            rank-based logging.
        logger_instance (Optional[Logger]): An existing Logger instance to use instead of creating
            a new one.

    Returns:
        callable: A decorator function.

    Usage:
        @log_prints(name="FunctionLogger", file_path="function_log.txt")
        def example_function():
            print("This will be logged")

        @log_prints(name="ClassLogger", file_path="class_log.txt")
        class ExampleClass:
            def __init__(self):
                print("This will be logged")

            def method(self):
                print("This will also be logged")
    """

    # from .pylightlog import Logger  # Import here to avoid circular imports

    def get_logger():
        return logger_instance or Logger(name=name,
                                         file_path=file_path,
                                         mode=mode,
                                         level=level,
                                         use_rank=use_rank,
                                         rank=rank,
                                         world_size=world_size,
                                         auto_detect_env=auto_detect_env)

    def decorator(obj: Callable) -> Any:
        if inspect.isclass(obj):
            # If decorating a class, wrap all methods
            for attr_name, attr_value in obj.__dict__.items():
                if callable(attr_value):
                    setattr(obj, attr_name, decorator(attr_value))
            return obj
        elif inspect.isfunction(obj) or inspect.ismethod(obj):
            # If decorating a function or method
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                logger = get_logger()
                with logger:
                    return obj(*args, **kwargs)

            return wrapper
        else:
            raise ValueError(
                "This decorator can only be applied to functions, methods, or classes.")

    return decorator
