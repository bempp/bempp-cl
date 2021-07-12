"""Interfaces to Helmholtz operators."""
import numpy as _np

from bempp.api.operators.boundary import common as _common
from bempp.api.assembly.boundary_operator import BoundaryOperator as _BoundaryOperator


def single_layer(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz single-layer boundary operator."""
    from .modified_helmholtz import single_layer as _modified_single_layer

    if _np.real(wavenumber) == 0:
        return _modified_single_layer(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    return _common.create_operator(
        "helmholtz_single_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_single_layer",
        "default_scalar",
        device_interface,
        precision,
        True,
    )


def double_layer(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz double-layer boundary operator."""
    from .modified_helmholtz import double_layer as _modified_double_layer

    if _np.real(wavenumber) == 0:
        return _modified_double_layer(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    return _common.create_operator(
        "helmholtz_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_double_layer",
        "default_scalar",
        device_interface,
        precision,
        True,
    )


def adjoint_double_layer(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz adj. double-layer boundary operator."""
    from .modified_helmholtz import (
        adjoint_double_layer as _modified_adjoint_double_layer,
    )

    if _np.real(wavenumber) == 0:
        return _modified_adjoint_double_layer(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    return _common.create_operator(
        "helmholtz_adjoint_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_adjoint_double_layer",
        "default_scalar",
        device_interface,
        precision,
        True,
    )


def hypersingular(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz hypersingular boundary operator."""
    from .modified_helmholtz import hypersingular as _hypersingular

    if domain.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Domain shapeset must be of type 'p1_discontinuous'.")

    if dual_to_range.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Dual to range shapeset must be of type 'p1_discontinuous'.")

    if _np.real(wavenumber) == 0:
        return _hypersingular(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    return _common.create_operator(
        "helmholtz_hypersingular_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_single_layer",
        "helmholtz_hypersingular",
        device_interface,
        precision,
        True,
    )


def multitrace_operator(
    grid,
    wavenumber,
    target=None,
    space_type="p1",
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """
    Simplified version of multitrace operator assembly.

    Parameters
    ----------
    grid : Grid
        Bempp grid object.
    wavenumber : complex
        A real or complex wavenumber
    target : Grid
        The grid for the range spaces. If target is None then
        target is set to the input grid (that is the domain
        grid).
    space_type : string
        Currently only "p1" is supported, which means
        that the operator is discretised with all P1 basis
        functions.
    parameters : Parameters
        An optional parameters object.
    assembler : string
        The assembler type.
    device_interface : DeviceInterface
        The device interface object to be used.
    precision : string
        Either "single" or "double" for single or
        double precision mode.

    Output
    ------
    The Helmholtz multitrace operator of the form
    [[-dlp, slp], [hyp, adj_dlp]], where
    dlp : double layer boundary operator
    slp : single layer boundary operator
    hyp : hypersingular boundary operator
    adj_dlp : adjoint double layer boundary operator.

    """
    import bempp.api
    from bempp.api.assembly.blocked_operator import BlockedOperator

    space = bempp.api.function_space(grid, "P", 1)

    if target is not None:
        target_space = bempp.api.function_space(target, "P", 1)
    else:
        target_space = space

    slp = single_layer(
        space,
        target_space,
        target_space,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    dlp = double_layer(
        space,
        target_space,
        target_space,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    hyp = hypersingular(
        space,
        target_space,
        target_space,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    adj_dlp = adjoint_double_layer(
        space,
        target_space,
        target_space,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    blocked = BlockedOperator(2, 2)

    blocked[0, 0] = -dlp
    blocked[0, 1] = slp
    blocked[1, 0] = hyp
    blocked[1, 1] = adj_dlp

    return blocked


def osrc_dtn(
    space,
    wavenumber,
    npade=2,
    theta=_np.pi / 3.0,
    damped_wavenumber=None,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the OSRC approximation to the DtN operator."""
    if space.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Space shapeset must be of type 'p1_discontinuous'.")

    return _OsrcDtN(
        space,
        parameters,
        [wavenumber, npade, theta, damped_wavenumber],
        device_interface,
        precision,
    )


class _OsrcDtN(_BoundaryOperator):
    """Implementation of the OSRC DtN operator."""

    def __init__(
        self, space, parameters, operator_options, device_interface=None, precision=None
    ):
        from bempp.api.operators import OperatorDescriptor

        super().__init__(space, space, space, parameters)

        self._device_interface = device_interface

        self._operator_descriptor = OperatorDescriptor(
            "osrc_dtn",
            operator_options,
            "laplace_beltrami",
            "default_sparse",
            precision,
            True,
            None,
            1,
        )

    @property
    def descriptor(self):
        """Operator descriptor."""
        return self._operator_descriptor

    def _assemble(self):
        """Assemble the operator."""
        from bempp.api.operators.boundary.sparse import identity
        from bempp.api.operators.boundary.sparse import laplace_beltrami
        from bempp.api.assembly.discrete_boundary_operator import (
            InverseSparseDiscreteBoundaryOperator,
        )

        space = self._domain
        wavenumber, npade, theta, damped_wavenumber = self.descriptor.options

        mass = identity(
            space,
            space,
            space,
            self._parameters,
            self._device_interface,
            self.descriptor.precision,
        ).weak_form()
        stiff = laplace_beltrami(
            space,
            space,
            space,
            self._parameters,
            self._device_interface,
            self.descriptor.precision,
        ).weak_form()

        if damped_wavenumber is None:
            bbox = space.grid.bounding_box
            rad = _np.linalg.norm(bbox[:, 1] - bbox[:, 0]) / 2.0
            dk = wavenumber + 0.4j * wavenumber ** (1.0 / 3.0) * rad ** (-2.0 / 3.0)
        else:
            dk = damped_wavenumber

        c0, alpha, beta = _pade_coeffs(npade, theta)

        series = c0 * mass
        for i in range(npade):
            element = (
                alpha[i]
                / (dk ** 2)
                * stiff
                * InverseSparseDiscreteBoundaryOperator(
                    mass - beta[i] / (dk ** 2) * stiff
                )
            )
            series -= element * mass
        operator = 1.0j * wavenumber * series

        return operator


def osrc_ntd(
    space,
    wavenumber,
    npade=2,
    theta=_np.pi / 3.0,
    damped_wavenumber=None,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the OSRC approximation to the NtD operator."""
    if space.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Space shapeset must be of type 'p1_discontinuous'.")

    return _OsrcNtD(
        space,
        parameters,
        [wavenumber, npade, theta, damped_wavenumber],
        device_interface,
        precision,
    )


class _OsrcNtD(_BoundaryOperator):
    """Implementation of the OSRC NtD operator."""

    def __init__(
        self, space, parameters, operator_options, device_interface=None, precision=None
    ):
        from bempp.api.operators import OperatorDescriptor

        super().__init__(space, space, space, parameters)

        self._device_interface = device_interface

        self._operator_descriptor = OperatorDescriptor(
            "osrc_ntd",
            operator_options,
            "laplace_beltrami",
            "default_sparse",
            precision,
            True,
            None,
            1,
        )

    @property
    def descriptor(self):
        """Operator descriptor."""
        return self._operator_descriptor

    def _assemble(self):
        from bempp.api.operators.boundary.sparse import identity
        from bempp.api.operators.boundary.sparse import laplace_beltrami
        from bempp.api.assembly.discrete_boundary_operator import (
            InverseSparseDiscreteBoundaryOperator,
        )

        space = self._domain
        wavenumber, npade, theta, damped_wavenumber = self.descriptor.options

        mass = identity(
            space,
            space,
            space,
            self._parameters,
            self._device_interface,
            self.descriptor.precision,
        ).weak_form()
        stiff = laplace_beltrami(
            space,
            space,
            space,
            self._parameters,
            self._device_interface,
            self.descriptor.precision,
        ).weak_form()

        if damped_wavenumber is None:
            bbox = space.grid.bounding_box
            rad = _np.linalg.norm(bbox[:, 1] - bbox[:, 0]) / 2.0
            dk = wavenumber + 0.4j * wavenumber ** (1.0 / 3.0) * rad ** (-2.0 / 3.0)
        else:
            dk = damped_wavenumber

        c0, alpha, beta = _pade_coeffs(npade, theta)

        series = c0 * mass
        for i in range(npade):
            element = (
                alpha[i]
                / (dk ** 2)
                * stiff
                * InverseSparseDiscreteBoundaryOperator(
                    mass - beta[i] / (dk ** 2) * stiff
                )
            )
            series -= element * mass
        operator = (
            1.0
            / (1.0j * wavenumber)
            * (
                mass
                * InverseSparseDiscreteBoundaryOperator(mass - 1.0 / (dk ** 2) * stiff)
                * series
            )
        )

        return operator


def _pade_coeffs(n, theta):
    """Compute the coefficients of the Pade series expansion."""
    aj = _np.zeros(n)
    bj = _np.zeros(n)
    for jj in range(1, n + 1):
        aj[jj - 1] = 2.0 / (2.0 * n + 1.0) * _np.sin(jj * _np.pi / (2.0 * n + 1.0)) ** 2
        bj[jj - 1] = _np.cos(jj * _np.pi / (2.0 * n + 1.0)) ** 2
    c0t = _np.exp(1.0j * theta / 2.0) * (
        1.0
        + _np.sum(
            (aj * (_np.exp(-1j * theta) - 1.0))
            / (1.0 + bj * (_np.exp(-1.0j * theta) - 1.0))
        )
    )
    ajt = (
        _np.exp(-1.0j * theta / 2.0)
        * aj
        / ((1.0 + bj * (_np.exp(-1.0j * theta) - 1.0)) ** 2)
    )
    bjt = _np.exp(-1.0j * theta) * bj / (1.0 + bj * (_np.exp(-1.0j * theta) - 1.0))

    return c0t, ajt, bjt


# def multitrace_operator(
# grid,
# wavenumber,
# segments=None,
# parameters=None,
# swapped_normals=None,
# assembler="dense_evaluator",
# device_interface=None,
# precision=None,
# ):
# """Assemble the Helmholtz multitrace operator."""
# from bempp.api.space import function_space
# from bempp.api.operators import _add_wavenumber
# from bempp.api.assembly.blocked_operator import GeneralizedBlockedOperator

# domain = function_space(
# grid,
# "P",
# 1,
# segments=segments,
# include_boundary_dofs=True,
# swapped_normals=swapped_normals,
# )
# range_ = domain
# dual_to_range = domain

# slp = single_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# dlp = double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# adlp = adjoint_double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# hyp = hypersingular(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# options = {"COMPLEX_KERNEL": None}

# _add_wavenumber(options, wavenumber)

# return GeneralizedBlockedOperator([[-dlp, slp], [hyp, adlp]])


# def transmission_operator(
# grid,
# wavenumber,
# rho_rel,
# refractive_index,
# segments=None,
# parameters=None,
# swapped_normals=None,
# assembler="dense_evaluator",
# device_interface=None,
# precision=None,
# ):
# """Assemble the Helmholtz transmission operator."""
# from bempp.api.space import function_space
# from bempp.api.operators import _add_wavenumber
# from bempp.api.assembly.blocked_operator import GeneralizedBlockedOperator


# wavenumber_int = wavenumber * refractive_index

# domain = function_space(
# grid,
# "P",
# 1,
# segments=segments,
# swapped_normals=swapped_normals,
# include_boundary_dofs=True,
# )
# range_ = domain
# dual_to_range = domain

# slp = single_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# slp_int = single_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# dlp = double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# dlp_int = double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# adlp = adjoint_double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# adlp_int = adjoint_double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# hyp = hypersingular(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# hyp_int = hypersingular(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# options = {"COMPLEX_KERNEL": None, "TRANSMISSION": None}

# _add_wavenumber(options, wavenumber)
# _add_wavenumber(options, rho_rel, "RHO_REL")
# _add_wavenumber(options, wavenumber_int, "WAVENUMBER_INT")

# return GeneralizedBlockedOperator(
# [
# [-dlp - dlp_int, slp + rho_rel * slp_int],
# [hyp + 1.0 / rho_rel * hyp_int, adlp + adlp_int],
# ]
# )
