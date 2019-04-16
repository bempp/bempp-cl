"Unit tests for PyOpenCL interface."

# pylint: disable = W0621

import numpy as np
import pytest

SHAPE = (1000, 1000)
DTYPE = "float32"


@pytest.fixture
def device_buffer():
    """Return a simple buffer."""
    from bempp.core.cl_helpers import default_device, DeviceBuffer

    return DeviceBuffer(SHAPE, DTYPE, default_device().context)


def test_buffer_mapping(device_buffer, device_interface):
    """Test if we can write to buffer and read from it."""
    np.random.seed(0)
    expected = np.random.rand(SHAPE[0], SHAPE[1])
    with device_buffer.host_array(device_interface, "write") as array:
        array[:] = expected

    with device_buffer.host_array(device_interface, "read") as array:
        actual = np.copy(array)

    np.testing.assert_allclose(actual, expected)
