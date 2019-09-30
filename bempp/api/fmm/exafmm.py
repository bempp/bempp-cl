"""Interface to ExaFMM."""
from .common import FmmInterface
import numpy as _np


class ExafmmLaplace(FmmInterface):
    """Interface to ExaFmm."""

    def __init__():
        """Initalize the ExaFmm Interface."""
        self._domain = None
        self._dual_to_range = None
        self._regular_order = None
        self._singular_order = None
        self._sources = None
        self._targets = None

        self._source_nodes = None
        self._target_nodes = None

        self._source_transform = None
        self._target_transform = None

    def setup(domain, dual_to_range, regular_order, singular_order, *args, **kwargs):
        """Setup the Fmm computation."""
        from .common import grid_to_points
        from bempp.api.integration.triangle_gauss import rule
        from exafmm_laplace import init_sources
        from exafmm_laplace import init_targets

        self.domain = domain
        self.dual_to_range = dual_to_range
        self._regular_order = regular_order
        self._singular_order = singular_order

        local_points = rule(regular_order)

        self._sources = grid_to_points(
            domain.grid, domain.support_elements, local_points
        )

        self._targets = grid_to_points(
            dual_to_range.grid, dual_to_range.support_elements, local_points
        )

        self._source_nodes = init_sources(
                self._sources, _np.zeros(len(self._sources), dtype=_np.float64))
        self._target_nodes = init_targets(
                self._targets)


