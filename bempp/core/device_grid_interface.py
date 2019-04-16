"""Provide an interface between a grid and a device."""


class DeviceGridInterface(object):
    """Administrates grid data on a given device."""

    def __init__(self, grid, device_interface, precision):
        """Initialize with a grid and a device."""
        from bempp.core.cl_helpers import DeviceBuffer
        from bempp.core.cl_helpers import get_type

        self._grid = grid
        self._device = device_interface

        self._buffer = DeviceBuffer.from_array(
            grid.as_array, device_interface, dtype=get_type(precision).real
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
        return self._buffer
