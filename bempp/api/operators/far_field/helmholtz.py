"""Helmholtz far-field operators."""


def single_layer(
    space, points, wavenumber, parameters=None, device_interface=None, precision=None
):
    """Return a Helmholtz far-field operator."""
    from bempp.core.dense_potential_assembler import DensePotentialAssembler
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.operators import _add_wavenumber
    from bempp.api.assembly.potential_operator import PotentialOperator

    options = {"KERNEL_FUNCTION": "helmholtz_single_layer_far_field"}
    _add_wavenumber(options, wavenumber)

    return PotentialOperator(
        DensePotentialAssembler(
            space,
            OperatorDescriptor(
                "helmholtz_single_layer_far_field", options, "helmholtz_scalar_far_field"
            ),
            points,
            1,
            True,
            device_interface,
            precision,
            parameters=parameters,
        )
    )


def double_layer(
    space, points, wavenumber, parameters=None, device_interface=None, precision=None
):
    """Return a Helmholtz double-layer far-field operator."""
    from bempp.core.dense_potential_assembler import DensePotentialAssembler
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.operators import _add_wavenumber
    from bempp.api.assembly.potential_operator import PotentialOperator

    options = {"KERNEL_FUNCTION": "helmholtz_double_layer_far_field"}
    _add_wavenumber(options, wavenumber)

    return PotentialOperator(
        DensePotentialAssembler(
            space,
            OperatorDescriptor(
                "helmholtz_double_layer_far_field", options, "helmholtz_scalar_far_field"
            ),
            points,
            1,
            True,
            device_interface,
            precision,
            parameters=parameters,
        )
    )
