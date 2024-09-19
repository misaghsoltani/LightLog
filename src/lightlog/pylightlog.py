import sys
from os import path as os_path
from typing import Optional

from .cpplightlog import CppLogger
from .levelsvalue import CRITICAL, DEBUG, ERROR, INFO, NOTSET, WARNING


class Logger(CppLogger):
    """
    A Python-friendly logger class that uses a C++-based logging core.

    This class handles Python-side logging by redirecting log messages to an
    underlying C++ logger (`CppLogger`). It supports logging to both files and
    stdout, optional rank-based logging for distributed systems, and redirection
    of `print()` statements to the logger.

    Attributes:
        name (str): The name of the logger instance, useful for identifying log sources.
        file_path (Optional[str]): The path to the log file. If `None`, logs are written
                                   to stdout.
        mode (str): The file mode used when writing logs to a file ('a' for append,
                    'w' for overwrite). Default is 'a'.
        level (int): The log level threshold, which controls which messages are logged.
                     Default is NOTSET (no formatting).
        use_rank (bool): Whether to include rank information in log messages. Useful
                         for multi-process or distributed environments.
        rank (Optional[int]): The process rank for distributed logging. Defaults to -1.
        world_size (Optional[int]): The total number of processes involved in distributed
                                    logging. Defaults to -1.
        auto_detect_env (Optional[str]): The environment auto-detection setting for rank
                                         information. Default is 'all'.
        original_stdout (TextIO): A reference to the original `sys.stdout`, used to restore
                                  standard output after `print()` redirection.
        _buffer (str): An internal buffer for storing partial log messages, allowing
                       line-based logging.

    Methods:
        __init__: Initializes the logger with the specified settings, including file
                  paths, and log levels.
        __del__: Ensures the logger is properly flushed and `print()` output is
                 reset to its original destination upon object deletion.
        write: Writes messages to the log, with optional arguments for including rank and
               specifying a new file path.
        flush: Flushes any buffered log messages.
        log: Logs messages with variable arguments,.
        info: Logs a message at the INFO level.
        debug: Logs a message at the DEBUG level.
        warning: Logs a message at the WARNING level.
        error: Logs a message at the ERROR level.
        critical: Logs a message at the CRITICAL level.
        fatal: Logs a message at the FATAL level.
        redirect_print: Redirects the Python's built-in `print()` function to output via
            the logger.
        reset_print: Restores the `print()` function to its original behavior.

    Example:
        >>> from lightlog import Logger
        >>> from logging import WARNING
        >>> logger = Logger('LogName', 'log.txt')
        >>> logger.log("Print to screen and save to 'log.txt'", level=WARNING)
        2024-09-8 17:34:16 | WARNING  | LogName | Print to screen and save to 'log.txt'
        >>> logger.log("Print to screen and save to 'log.txt'")
        Print to screen and save to 'log.txt'
        >>> logger.info("Print to screen and save to 'log.txt'")
        2024-09-8 17:34:48 |   INFO   | LogName | Print to screen and save to 'log.txt'
        >>> # Logging example for distributed computing environments with mnauall rank setting
        >>> logger = Logger('LogName', 'log.txt', use_rank=True, rank=0, world_size=10)
        >>> logger.log("Print to screen and save to 'log.txt'", level=WARNING)
        [0/10] 2024-09-8 17:36:43 | WARNING  | LogName | Print to screen and save to 'log.txt'
        >>> logger.log("Print to screen and save to 'log.txt'")
        [0/10] Print to screen and save to 'log.txt'
        >>> # Logging example for distributed computing environments with auto rank detection
        >>> # Let's assume the environment is 'mpirun'
        >>> # and the environment variable 'OMPI_COMM_WORLD_SIZE' is set to 34
        >>> # and the environment variable 'OMPI_COMM_WORLD_RANK' is set to 8
        >>> logger = Logger('LogName', 'log.txt', use_rank=True)
        >>> logger.log("Print to screen and save to 'log.txt'")
        [8/34] Print to screen and save to 'log.txt'
        >>> logger.info("Print to screen and save to 'log.txt'")
        [8/34] 2024-09-8 17:37:50 |   INFO   | LogName | Print to screen and save to 'log.txt'
        >>> # Logging example for distributed computing environments with given auto_detect_env
        >>> # Let's assume the environment is 'torchrun' and the rank is 8 and the world size is 34
        >>> # i.e. the environment variable 'RANK' is set to 8
        >>> # and the environment variable 'WORLD_SIZE' is set to 34
        >>> logger = Logger('LogName', 'log.txt', use_rank=True, auto_detect_env='torchrun')
        >>> logger.critical("Print to screen and save to 'log.txt'")
        [8/34] 2024-09-8 17:38:56 | CRITICAL | LogName | Print to screen and save to 'log.txt'
        >>> # Redirecting print to the logger
        >>> logger = Logger('LogName', 'log.txt')
        >>> print("Print to the screen only")
        Print to the screen only
        >>> logger.redirect_print()
        >>> print("Print to screen and save to 'log.txt'")
        Print to screen and save to 'log.txt'
        >>> logger.reset_print()
        >>> print("Print to the screen only")
        Print to the screen only
        >>> # If the use_rank is set to True, the rank and world size label will be added
    """

    def __init__(self,
                 name: str,
                 file_path: Optional[str] = None,
                 mode: str = 'a',
                 level: int = NOTSET,
                 use_rank: bool = False,
                 rank: Optional[int] = None,
                 world_size: Optional[int] = None,
                 auto_detect_env: Optional[str] = None,
                 log_rank: Optional[int] = None) -> None:
        """
        Initializes the Logger instance, setting up logging parameters and configuring
        the output destination.

        Args:
            name (str): The name of the logger instance.
            file_path (Optional[str]): Path to the log file. If `None`, logs will be
                                       written to stdout only.
            mode (str): File mode for writing logs, either 'a' (append) or 'w'
                        (overwrite). Default is 'a'.
            level (int, optional): The log level. Can be one of `CRITICAL`, `ERROR`, `WARNING`,
                                   `INFO`, `DEBUG`. Default is -1 (no level filtering).
            use_rank (bool, optional): If `True`, includes process rank in the logs. This is
                                       useful in distributed systems. Default is `False`.
            rank (Optional[int]): The rank of the process in a distributed system.
                                  If `None`, it uses the rank auto-detection. Default is `None`.
            world_size (Optional[int]): The total number of processes in a distributed system.
                                        If `None`, it uses the rank auto-detection.
                                        Default is `None`.
            auto_detect_env (Optional[str]): Specifies how to automatically detect the
                                             environment for rank-based logging. Default is 'all'.
                                             'all' will try to use the environment variables from
                                             the related variable for the environments:
                                             ['mpirun', 'torchrun', 'horovodrun', 'slurm', 'nccl'].
                                             or the general case of using `RANK` and `WORLD_SIZE`.
            log_rank (Optional[int]): The rank of the process on which to log/print. Default
                                      is `None`.

        Raises:
            ValueError: Raised if an invalid file mode is provided.
            IOError: Raised if the file specified by `file_path` cannot be opened for writing.

        Example:
            >>> logger = Logger(name="app_logger", file_path="application.log")
        """
        self.original_stdout = sys.stdout  # Original stdout saved for resetting later
        self.use_rank = use_rank
        self.name = name
        self.file_path = os_path.abspath(file_path) if file_path else ''
        self.mode = mode
        self.level = level
        self.rank = rank or -1
        self.world_size = world_size or -1
        self.auto_detect_env = auto_detect_env or 'all'
        self._buffer = ""  # Buffer for handling incomplete log messages
        self.log_rank = log_rank or -1

        # Call the base CppLogger constructor
        super().__init__(name, self.file_path, self.mode, self.level, self.use_rank, self.rank,
                         self.world_size, self.auto_detect_env, self.log_rank)

    def __del__(self) -> None:
        """
        Destructor that ensures buffered logs are flushed and the logger is closed.

        This method is called when the Logger object is about to be destroyed.
        It ensures that any remaining buffered logs are written and the logger
        is properly closed.
        """

        self.flush()
        self.close()
        self.reset_print()

    def write(self,
              message: str,
              level: int = NOTSET,
              use_rank: bool = False,
              new_file_path: str = None) -> None:
        """
        Writes a message to the logger with optional log level, rank, and file path settings.

        This method appends the incoming `message` to an internal buffer, splitting it
        into lines when necessary. Complete lines are passed to the underlying logging
        core (`CppLogger`) with the specified `level`, `use_rank`, and `new_file_path`.

        The method ensures that partially written lines (i.e., those not ending with
        a newline character) remain in the buffer until they are completed, allowing for
        proper handling of streaming input.

        Args:
            message (str): The message to be logged. If the message contains multiple lines,
                           each line will be processed and logged individually.
            level (int, optional): The log level for the message. Default is -1, which
                                   means no specific log level is used.
            use_rank (bool, optional): If `True`, include rank information (useful in
                                       distributed or multiprocess environments). Defaults
                                       to `False`.
            new_file_path (str, optional): If provided, logs the message to a different file
                                           than the original file specified during initialization.
                                           Defaults to None (i.e., no file redirection).

        Example:
            >>> logger.write("Starting the process...", level=logging.DEBUG)

        Behavior:
            - In case the message is split into multiple lines, each line will be logged
            separately, and only the last incomplete line (if any) will remain in the buffer.
            - The `level` and `use_rank` parameters can be customized for each `write()` call.

        Warning:
            - Make sure to `flush()` the logger to ensure all buffered messages are written
            before terminating the application or closing the logger.
        """
        self._buffer += message

        # Split the buffer into lines, leaving incomplete lines in the buffer
        lines = self._buffer.splitlines(keepends=True)
        if not self._buffer.endswith('\n'):
            self._buffer = lines.pop()  # Keep incomplete line in buffer
        else:
            self._buffer = ""

        # Write each complete line to the log
        for line in lines:
            new_file_path = os_path.abspath(new_file_path) if new_file_path else ''
            super().log(line, level, self.use_rank or use_rank, new_file_path)

    def flush(self) -> None:
        """
        Flushes the internal message buffer to the log output.

        This method ensures that any incomplete message remaining in the buffer
        is logged. After the buffer is flushed, it is cleared, and the underlying
        logger's `flush()` method is called to ensure all messages are properly
        written to the log destination (e.g file and console).

        This is particularly useful when dealing with partially completed log
        messages that may not have ended with a newline character (`\\n`). Calling
        `flush()` guarantees that such messages will be immediately logged.

        Example:
            >>> logger.write("This is a partial message")
            >>> logger.flush()  # Ensures the partial message is logged

        Behavior:
            - If the buffer contains an incomplete line, it will be flushed (i.e., logged)
            immediately, even if it does not end with a newline character.
            - After flushing, the buffer is reset and the logger state is updated.
        """
        if self._buffer:  # If buffer is not empty
            super().log(self._buffer, level=-1, use_rank=self.use_rank)
            self._buffer = ""
        super().flush()

    def log(self,
            *args: object,
            sep: Optional[str] = " ",
            end: Optional[str] = "\n",
            level: int = NOTSET,
            use_rank: bool = False,
            new_file_path: str = None) -> None:
        """
        Logs a formatted message by joining multiple arguments.

        This method allows for logging arbitrary numbers of arguments by converting
        them into strings, joining them with the specified separator (`sep`), and
        appending the specified ending (`end`). The resulting message is passed
        to the logger's C++ core with optional log level, rank, and file path settings.

        Args:
            *args (object): The components of the message to be logged. Each argument
                            will be converted to a string and joined using the specified
                            `sep`.
            sep (str, optional): Separator used to join the arguments. Defaults to a space `" "`.
            end (str, optional): String appended after the joined message. Defaults to a
                newline `"\n"`.
            level (int, optional): The log level for the message. Defaults to NOTSET
            use_rank (bool, optional): If `True`, includes rank information. Defaults to `False`.
            new_file_path (str, optional): If provided, logs the message to a different file than
                                        the one specified when initializing the logger. Defaults to
                                        None.

        Example:
            >>> logger.log("Initializing module:", "ModuleA", sep=" ", end=".\n")
            >>> logger.log("Value:", 42, "Threshold:", 100, sep=", ")

        Behavior:
            - Converts all arguments to strings, joins them with the specified separator (`sep`),
            and appends the string `end`.
            - The `level` and `use_rank` can be specified dynamically for each call, allowing
            flexibility in logging different levels of messages in different contexts.
            - Supports redirection to a new file via `new_file_path` if needed.
        """
        message = sep.join(map(str, args)) + end
        new_file_path = os_path.abspath(new_file_path) if new_file_path else ''
        super().log(message, level, self.use_rank or use_rank, new_file_path)

    def reconfigure(self,
                    name: str = None,
                    new_file_path: Optional[str] = None,
                    mode: str = 'a',
                    level: int = None,
                    use_rank: bool = None,
                    rank: Optional[int] = None,
                    world_size: Optional[int] = None,
                    auto_detect_env: Optional[str] = None,
                    log_rank: Optional[int] = None) -> None:
        """
        Reconfigures the logger with new settings, updating all relevant parameters.

        Args:
            name (str): The name of the logger instance. If None, the previous name setting
                will be retained.
            new_file_path (Optional[str]): Path to the log file. If None, the previous file
                setting will be retained.
            mode (str): File mode for writing logs, either 'a' (append) or 'w' (overwrite).
                Defaults to 'a'.
            level (int): The log level. Can be one of CRITICAL, ERROR, WARNING, INFO, DEBUG.
                Defaults to None.
            use_rank (bool): If True, includes process rank in the logs. This is useful in
                distributed systems. Defaults to None.
            rank (Optional[int]): The rank of the process in a distributed system. If None, it
                uses the rank auto-detection. Defaults to None.
            world_size (Optional[int]): The total number of processes in a distributed system.
                If None, it uses the rank auto-detection. Defaults to None.
            auto_detect_env (Optional[str]): Specifies how to automatically detect the environment
                for rank-based logging. Defaults to None.
            log_rank (Optional[int]): The rank of the process on which to log/print. Defaults
                to None.

        Raises:
            ValueError: If an invalid file mode is provided.
            IOError: If the file specified by new_file_path cannot be opened for writing.

        Examples:
            >>> logger = Logger(name="app_logger", file_path="application.log")
            >>> logger.reconfigure(name="new_logger", new_file_path="new_application.log")
        """
        self.use_rank = use_rank or self.use_rank
        self.name = name or self.name
        self.file_path = os_path.abspath(new_file_path) if new_file_path else self.file_path
        self.mode = mode or self.mode
        self.level = level or self.level
        self.rank = rank or self.rank
        self.world_size = world_size or self.world_size
        self.auto_detect_env = auto_detect_env or self.auto_detect_env
        self.log_rank = log_rank or self.rank

        self.flush()
        # Call the base CppLogger constructor
        super().reconfigure(self.name, self.file_path, self.mode, self.level, self.use_rank,
                            self.rank, self.world_size, self.auto_detect_env, self.log_rank)

    def info(self,
             *args: object,
             sep: Optional[str] = " ",
             end: Optional[str] = "\n",
             use_rank: bool = False,
             new_file_path: str = None) -> None:
        """
        Logs an INFO level message.

        Args:
            *args: The message components to be joined and logged.
            sep: Optional; Separator used to join the components. Default is a space.
            end: Optional; String appended after the message. Default is newline.
            use_rank: Optional; If True, include rank information. Default is False.
            new_file_path: Optional; Write to a new log file if provided.

        Example:
            >>> logger.info("This is an info message.")
        """
        new_file_path = os_path.abspath(new_file_path) if new_file_path else ''
        self.log(*args,
                 sep=sep,
                 end=end,
                 level=INFO,
                 use_rank=use_rank,
                 new_file_path=new_file_path)

    def debug(self,
              *args: object,
              sep: Optional[str] = " ",
              end: Optional[str] = "\n",
              use_rank: bool = False,
              new_file_path: str = None) -> None:
        """
        Logs a DEBUG level message.

        Args:
            *args: The message components to be joined and logged.
            sep: Optional; Separator used to join the components. Default is a space.
            end: Optional; String appended after the message. Default is newline.
            use_rank: Optional; If True, include rank information. Default is False.
            new_file_path: Optional; Write to a new log file if provided.

        Example:
            >>> logger.debug("This is a debug message.")
        """
        new_file_path = os_path.abspath(new_file_path) if new_file_path else ''
        self.log(*args,
                 sep=sep,
                 end=end,
                 level=DEBUG,
                 use_rank=use_rank,
                 new_file_path=new_file_path)

    def warning(self,
                *args: object,
                sep: Optional[str] = " ",
                end: Optional[str] = "\n",
                use_rank: bool = False,
                new_file_path: str = None) -> None:
        """
        Logs a WARNING level message.

        Args:
            *args: The message components to be joined and logged.
            sep: Optional; Separator used to join the components. Default is a space.
            end: Optional; String appended after the message. Default is newline.
            use_rank: Optional; If True, include rank information. Default is False.
            new_file_path: Optional; Write to a new log file if provided.


        Example:
            >>> logger.warning("This is a warning message.")
        """

        new_file_path = os_path.abspath(new_file_path) if new_file_path else ''
        self.log(*args,
                 sep=sep,
                 end=end,
                 level=WARNING,
                 use_rank=use_rank,
                 new_file_path=new_file_path)

    def error(self,
              *args: object,
              sep: Optional[str] = " ",
              end: Optional[str] = "\n",
              use_rank: bool = False,
              new_file_path: str = None) -> None:
        """
        Logs an ERROR level message.

        Args:
            *args: The message components to be joined and logged.
            sep: Optional; Separator used to join the components. Default is a space.
            end: Optional; String appended after the message. Default is newline.
            use_rank: Optional; If True, include rank information. Default is False.
            new_file_path: Optional; Write to a new log file if provided.


        Example:
            >>> logger.error("This is a warning message.")
        """

        new_file_path = os_path.abspath(new_file_path) if new_file_path else ''
        self.log(*args,
                 sep=sep,
                 end=end,
                 level=ERROR,
                 use_rank=use_rank,
                 new_file_path=new_file_path)

    def critical(self,
                 *args: object,
                 sep: Optional[str] = " ",
                 end: Optional[str] = "\n",
                 use_rank: bool = False,
                 new_file_path: str = None) -> None:
        """
        Logs a CRITICAL level message.

        Args:
            *args: The message components to be joined and logged.
            sep: Optional; Separator used to join the components. Default is a space.
            end: Optional; String appended after the message. Default is newline.
            use_rank: Optional; If True, include rank information. Default is False.
            new_file_path: Optional; Write to a new log file if provided.


        Example:
            >>> logger.critical("This is a warning message.")
        """

        new_file_path = os_path.abspath(new_file_path) if new_file_path else ''
        self.log(*args,
                 sep=sep,
                 end=end,
                 level=CRITICAL,
                 use_rank=use_rank,
                 new_file_path=new_file_path)

    def redirect_print(self) -> None:
        """
        Redirects the built-in `print()` function's output to the logger instance.

        This method replaces the standard output (`sys.stdout`) with the logger
        itself, meaning all future calls to `print()` will be captured and logged
        according to the logger's configuration, including logging levels, file
        output, and rank-based logging if applicable.

        Note:
            - Logging will respect the settings defined during the logger's
            initialization (such as log level, rank usage, etc.).
            - `reset_print()` can be called to restore the original `print()` behavior.

        Example:
            >>> logger = Logger(name="example_logger", file_path="log.txt", use_rank=True)
            >>> logger.redirect_print()
            >>> print("This will be printed to console and saved to 'log.txt'.")

        """
        self.flush()
        sys.stdout = self

    def reset_print(self) -> None:
        """
        Restores the built-in `print()` function's output to the original `sys.stdout`.

        This method undoes the redirection initiated by `redirect_print()`, restoring
        the standard output stream. Any subsequent calls to `print()` will behave
        normally, outputting to the console or any other output stream that was
        originally assigned to `sys.stdout`.

        Example:
            >>> logger = Logger(name="example_logger", file_path="log.txt"O)
            >>> logger.redirect_print()
            >>> print("This goes to the log.")
            >>> logger.reset_print()
            >>> print("This goes back to the console.")
        """
        self.flush()
        sys.stdout = self.original_stdout

    def close(self) -> None:
        """
        Closes the logger and the file, flushing any remaining messages and restoring the original
        `sys.stdout` behavior.

        This method should be called before the logger instance is deleted or goes out
        of scope to ensure that all buffered messages are written to the log file and
        that the `print()` function is reset to its original behavior.

        Example:
            >>> logger = Logger(name="example_logger", file_path="log.txt")
            >>> logger.close()
        """
        self.flush()
        self.reset_print()
        super().close()

    def __enter__(self):
        """
        Enter the runtime context related to this object.

        This method allows the Logger to be used as a context manager,
        redirecting all print statements within the 'with' block to the logger.

        Returns:
            self: The Logger instance.
        """
        self.redirect_print()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context related to this object.

        This method restores the original print behavior when exiting the 'with' block.

        Args:
            exc_type: The exception type if an exception was raised in the 'with' block.
            exc_value: The exception value if an exception was raised in the 'with' block.
            traceback: The traceback if an exception was raised in the 'with' block.
        """
        self.reset_print()
