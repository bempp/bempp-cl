"""Global initialization for Bempp."""

import os as _os
import tempfile as _tempfile
import logging as _logging
import time as _time

from bempp.api.utils import DefaultParameters
from bempp.api.utils.helpers import MemProfiler

from bempp.api.utils.helpers import assign_parameters
from bempp.api.grid.io import import_grid
from bempp.api.grid.io import export
from bempp.api.grid.grid import Grid
from bempp.api.assembly.grid_function import GridFunction
from bempp.api.assembly.grid_function import real_callable
from bempp.api.assembly.grid_function import complex_callable

from bempp.api.space import function_space

from bempp.api import shapes
from bempp.api import integration
from bempp.api import operators
from bempp.api.linalg.direct_solvers import lu, compute_lu_factors
from bempp.api.linalg.iterative_solvers import gmres, cg
from bempp.api.assembly.discrete_boundary_operator import as_matrix
from bempp.api.assembly.boundary_operator import ZeroBoundaryOperator
from bempp.api.assembly.boundary_operator import MultiplicationOperator
from bempp.api.assembly.blocked_operator import BlockedOperator
from bempp.api.assembly.blocked_operator import GeneralizedBlockedOperator

from bempp.api.utils import pool
from bempp.api.utils.pool import create_device_pool

# Disable Numba warnings


from numba.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaPerformanceWarning,
)
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


CONSOLE_LOGGING_HANDLER = None
DEFAULT_LOGGING_FORMAT = "%(name)s:%(levelname)s: %(message)s"
DEBUG = _logging.DEBUG
TIMING = 11
INFO = _logging.INFO
WARNING = _logging.WARNING
ERROR = _logging.ERROR
CRITICAL = _logging.CRITICAL

LOG_LEVEL = {
    "debug": DEBUG,
    "timing": TIMING,
    "info": INFO,
    "warning": WARNING,
    "error": ERROR,
    "critical": CRITICAL,
}

GLOBAL_PARAMETERS = DefaultParameters()


def _init_logger():
    """Initialize the Bempp logger."""

    _logging.addLevelName(11, "TIMING")
    logger = _logging.getLogger("bempp")
    logger.setLevel(DEBUG)
    logger.addHandler(_logging.NullHandler())
    return logger


def log(message, level="info", flush=True):
    """Log including default flushing for IPython."""
    LOGGER.log(LOG_LEVEL[level], message)
    if flush:
        flush_log()


def flush_log():
    """Flush all handlers. Necessary for Jupyter."""
    for handler in LOGGER.handlers:
        handler.flush()


def enable_console_logging(level="info"):
    """Enable console logging and return the console handler."""
    from bempp.api.utils import pool

    # pylint: disable=W0603
    global CONSOLE_LOGGING_HANDLER
    if not CONSOLE_LOGGING_HANDLER:
        console_handler = _logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL[level])
        if pool.is_worker():
            console_handler.setFormatter(
                _logging.Formatter(
                    f"%(name)s:PROC{pool._MY_ID}:%(levelname)s: %(message)s", "%H:%M:%S"
                )
            )
        else:
            console_handler.setFormatter(
                _logging.Formatter(
                    "%(name)s:HOST:%(levelname)s: %(message)s", "%H:%M:%S"
                )
            )
        LOGGER.addHandler(console_handler)
        CONSOLE_LOGGING_HANDLER = console_handler
    return CONSOLE_LOGGING_HANDLER


# def enable_file_logging(file_name, level=DEBUG, logging_format=DEFAULT_LOGGING_FORMAT):
# """Enable logging to a specific file."""

# file_handler = _logging.FileHandler(file_name)
# file_handler.setLevel(level)
# file_handler.setFormatter(_logging.Formatter(logging_format, "%H:%M:%S"))
# LOGGER.addHandler(file_handler)
# return file_handler


def set_logging_level(level):
    """Set the logging level."""
    LOGGER.setLevel(LOG_LEVEL[level])


# pylint: disable=too-few-public-methods
class Timer:
    """Context manager to measure time in Bempp."""

    def __init__(self, enable_log=True, message="", level="timing"):
        """Constructor."""
        self.start = 0
        self.end = 0
        self.interval = 0
        self.enable_log = enable_log
        self.level = level
        self.message = message

    def __enter__(self):
        if self.enable_log:
            log("Start operation: " + self.message, level=self.level)
        self.start = _time.time()
        return self

    def __exit__(self, *args):
        self.end = _time.time()
        self.interval = self.end - self.start
        if self.enable_log:
            log("Finished Operation: " + self.message + f": {self.interval}s", level=self.level)


def test(precision="double", vectorization="auto"):
    """ Runs Bempp python unit tests """
    import pytest

    options = []

    options.append("--precision=" + precision)
    options.append("--vec=" + vectorization)
    options.append(BEMPP_PATH)

    pytest.main(options)


def benchmark(precision="double", vectorization="auto", capture_output=True):
    """Run py.test benchmarks"""
    import pytest

    benchmark_dir = _os.path.join(BEMPP_PATH, "./benchmarks/")
    options = [
        "-o",
        "python_files=benchmark_*.py",
        "-o",
        "python_classes=Benchmark",
        "-o",
        "python_functions=*_benchmark",
        "-p",
        "no:warnings",
    ]

    options.append("--precision=" + precision)
    options.append("--vec=" + vectorization)

    if not capture_output:
        options.append("-s")

    options.append(benchmark_dir)

    pytest.main(options)


LOGGER = _init_logger()

BEMPP_PATH = _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.realpath(__file__)), "..")
)

# pylint: disable=W0702
# try:
#    if _os.environ['BEMPP_CONSOLE_LOGGING'] == '1':
#        enable_console_logging()
# except:
#    pass

TMP_PATH = _tempfile.mkdtemp()

# Get the path to Gmsh


def _gmsh_path():
    """Find Gmsh."""
    from bempp.api.utils import which

    if _os.name == "nt":
        gmp = which("gmsh.exe")
    else:
        gmp = which("gmsh")
    if gmp is None:
        print(
            "Could not find Gmsh."
            + "Interactive plotting and shapes module not available."
        )
    return gmp


def _get_version():
    """Get version string."""
    from bempp import version

    return version.__version__


GMSH_PATH = _gmsh_path()

__version__ = _get_version()

PLOT_BACKEND = "jupyter_notebook"
USE_JIT = True

DEFAULT_DEVICE_INTERFACE = "opencl"
DEFAULT_PRECISION = "double"
VECTORIZATION_MODE = 'auto'


ALL = -1  # Useful global identifier
