"""Provide an interface between a grid and a device."""
import numpy as _np

class DeviceGridInterface(object):
    """Administrates grid data on a given device."""

    def __init__(self, grid, device_interface, precision):
        """Initialize with a grid and a device."""
        from bempp.core.cl_helpers import DeviceBuffer
        from bempp.core.cl_helpers import get_type

        self._grid = grid
        self._device = device_interface

        self._grid_buffer = DeviceBuffer.from_array(
            grid.as_array, device_interface, dtype=get_type(precision).real
        )

        self._elements_buffer = DeviceBuffer.from_array(
                grid.elements, device_interface, dtype=_np.uint32,
                access_mode="read_only",
                order="F",
                )

    @property
    def grid(self):
        """Return grid."""
        return self._grid

    @property
    def device(self):
        """Return device."""
        return self._device

    @property
    def buffer(self):
        """Return device buffer."""
        return self._grid_buffer

    @property
    def grid_buffer(self):
        """Return grid buffer."""
        return self._grid_buffer

    @property
    def elements_buffer(self):
        """Return elements buffer."""
        return self._elements_buffer
