"Define fixtures and other test helpers."

import os

import numpy as np
import pytest

import bempp.api
import bempp.core


def pytest_addoption(parser):
    parser.addoption(
        "--vec",
        action="store",
        default="auto",
        help="Valid values: auto novec vec4 vec8 vec16",
    )
    parser.addoption(
        "--precision",
        action="store",
        default="double",
        help="Valid values: single double",
    )
    parser.addoption(
        "--device", action="store", default="numba", help="Valid values: numba opencl",
    )


@pytest.fixture()
def device_interface(request):
    value = request.config.getoption("--device")
    if value not in ["numba", "opencl"]:
        raise ValueError("device must be one of: 'numba', 'opencl'")
    return value


# @pytest.fixture(scope="session", autouse=True)
# def set_device_options(request):
# """Set device options."""
# vec_mode = request.config.getoption("--vec")
# if not vec_mode in ["auto", "novec", "vec4", "vec8", "vec16"]:
# raise ValueError(
# "vec must be one of: 'auto', 'novec', 'vec4', 'vec8', 'vec16'"
# )
# bempp.api.VECTORIZATION = vec_mode


@pytest.fixture()
def precision(request):
    """Return precision."""
    value = request.config.getoption("--precision")
    if value not in ["single", "double"]:
        raise ValueError("precision must be one of: 'single', 'double'")
    return value


@pytest.fixture
def two_element_grid():
    """Simple grid consisting of two elements."""
    from bempp.api.grid import Grid

    vertices = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])

    elements = np.array([[0, 1], [1, 2], [3, 3]])

    return Grid(vertices, elements)


@pytest.fixture
def small_sphere(helpers):
    """A regular sphere with 512 elements."""
    return helpers.load_grid("small_sphere")


@pytest.fixture
def default_parameters():
    """Return a default parameters object."""
    from bempp.api.utils.parameters import DefaultParameters

    parameters = DefaultParameters()
    parameters.quadrature.regular = 4
    parameters.quadrature.singular = 4

    return parameters


@pytest.fixture
def small_piecewise_const_space(small_sphere):
    """A simple piecewise constant space on a small sphere."""
    return bempp.api.function_space(small_sphere, "DP", 0)


@pytest.fixture
def laplace_slp_small_sphere(small_piecewise_const_space, default_parameters):
    """The Laplace single layer operator on a small sphere."""
    from bempp.api.operators.boundary.laplace import single_layer

    return single_layer(
        small_piecewise_const_space,
        small_piecewise_const_space,
        small_piecewise_const_space,
    )


@pytest.fixture
def helpers():
    """Return static Helpers class."""
    return Helpers


class Helpers(object):
    """Helper class with static helper methods for unit tests."""

    @staticmethod
    def load_npy_data(name):
        """Load data stored as npy file (give name without .npy)."""
        data_name = os.path.join(
            Helpers.bempp_path(), "bempp/test_data/" + name + ".npy"
        )
        return np.load(data_name)

    @staticmethod
    def load_npz_data(name):
        """Load data stored as npz file (give name without .npz)."""
        data_name = os.path.join(
            Helpers.bempp_path(), "bempp/test_data/" + name + ".npz"
        )
        return np.load(data_name)

    @staticmethod
    def load_grid(name):
        """Load grid stored as msh file (give name without ending)"""
        grid_name = os.path.join(
            Helpers.bempp_path(), "bempp/test_data/" + name + ".msh"
        )
        return bempp.api.import_grid(grid_name)

    @staticmethod
    def bempp_path():
        """Return path of the bempp module."""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    @staticmethod
    def default_tolerance(precision):
        """Given a precision return default tolerance."""
        if precision == "single":
            return 5e-4
        if precision == "double":
            return 1e-10
