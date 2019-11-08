"""Global initialization for Bempp."""

import os as _os
import tempfile as _tempfile
import logging as _logging
import time as _time

import pyopencl as _cl
from bempp.api.utils import DefaultParameters
from bempp.api.utils.helpers import MemProfiler

from bempp.api.utils.helpers import assign_parameters
from bempp.api.grid.io import import_grid
from bempp.api.grid.io import export
from bempp.api.assembly.grid_function import GridFunction
from bempp.api.assembly.grid_function import real_callable
from bempp.api.assembly.grid_function import complex_callable
from bempp.core.cl_helpers import DeviceInterface as _DeviceInterface
from bempp.core.cl_helpers import show_available_platforms_and_devices
from bempp.core.cl_helpers import set_default_device
from bempp.core.cl_helpers import default_device
from bempp.core.cl_helpers import default_context
from bempp.core.cl_helpers import get_precision

from bempp.api.space import function_space

from bempp.api import shapes
from bempp.api import integration
from bempp.api import operators
from bempp.api.linalg.direct_solvers import lu, compute_lu_factors
from bempp.api.linalg.iterative_solvers import gmres, cg
from bempp.api.assembly.discrete_boundary_operator import as_matrix
from bempp.api.assembly.boundary_operator import ZeroBoundaryOperator
from bempp.api.assembly.boundary_operator import MultiplicationOperator

# Disable Numba warnings


from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


CONSOLE_LOGGING_HANDLER = None
DEFAULT_LOGGING_FORMAT = "%(name)s:%(levelname)s: %(message)s"
DEBUG = _logging.DEBUG
INFO = _logging.INFO
WARNING = _logging.WARNING
ERROR = _logging.ERROR
CRITICAL = _logging.CRITICAL

GLOBAL_PARAMETERS = DefaultParameters()


def _init_logger():
    """Initialize the Bempp logger."""

    logger = _logging.getLogger()
    logger.setLevel(INFO)
    logger.addHandler(_logging.NullHandler())
    return logger


def log(message, level=INFO, flush=True):
    """Log including default flushing for IPython."""
    LOGGER.log(level, message)
    if flush:
        flush_log()


def flush_log():
    """Flush all handlers. Necessary for Jupyter."""
    for handler in LOGGER.handlers:
        handler.flush()


def enable_console_logging(level=DEBUG):
    """Enable console logging and return the console handler."""
    # pylint: disable=W0603
    global CONSOLE_LOGGING_HANDLER
    if not CONSOLE_LOGGING_HANDLER:
        console_handler = _logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(
            _logging.Formatter(DEFAULT_LOGGING_FORMAT, "%H:%M:%S")
        )
        LOGGER.addHandler(console_handler)
        CONSOLE_LOGGING_HANDLER = console_handler
    return CONSOLE_LOGGING_HANDLER


def enable_file_logging(file_name, level=DEBUG, logging_format=DEFAULT_LOGGING_FORMAT):
    """Enable logging to a specific file."""

    file_handler = _logging.FileHandler(file_name)
    file_handler.setLevel(level)
    file_handler.setFormatter(_logging.Formatter(logging_format, "%H:%M:%S"))
    LOGGER.addHandler(file_handler)
    return file_handler


def set_logging_level(level):
    """Set the logging level."""
    LOGGER.setLevel(level)


def timeit(message):
    """Decorator to time a method in Bempp"""

    def timeit_impl(fun):
        """Implementation of timeit."""

        def timed_fun(*args, **kwargs):
            """The actual timer function."""
            if not GLOBAL_PARAMETERS.verbosity.extended_verbosity:
                return fun(*args, **kwargs)

            start_time = _time.time()
            res = fun(*args, **kwargs)
            end_time = _time.time()
            log(message + " : {0:.3e}s".format(end_time - start_time))
            return res

        return timed_fun

    return timeit_impl


# pylint: disable=too-few-public-methods
class Timer:
    """Context manager to measure time in BEM++."""

    def __init__(self):
        """Constructor."""
        self.start = 0
        self.end = 0
        self.interval = 0

    def __enter__(self):
        self.start = _time.time()
        return self

    def __exit__(self, *args):
        self.end = _time.time()
        self.interval = self.end - self.start


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


GMSH_PATH = _gmsh_path()

VECTORIZATION = "auto"
DEVICE_PRECISION_CPU = "double"
DEVICE_PRECISION_GPU = "single"
PLOT_BACKEND = "gmsh"

ALL = -1  # Useful global identifier
