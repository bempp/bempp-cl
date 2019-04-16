"""Laplace potential operators."""


def single_layer(space, points, parameters=None, device_interface=None, precision=None):
    """Return a Laplace single-layer potential operator."""
    from bempp.core.dense_potential_assembler import DensePotentialAssembler
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator

    return PotentialOperator(
        DensePotentialAssembler(
            space,
            OperatorDescriptor(
                "laplace_single_layer",
                {"KERNEL_FUNCTION": "laplace_single_layer"},
                "default_dense",
            ),
            points,
            1,
            False,
            device_interface,
            precision,
            parameters=parameters,
        )
    )


def double_layer(space, points, parameters=None, device_interface=None, precision=None):
    """Return a Laplace double-layer potential operator."""
    from bempp.core.dense_potential_assembler import DensePotentialAssembler
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.assembly.potential_operator import PotentialOperator

    return PotentialOperator(
        DensePotentialAssembler(
            space,
            OperatorDescriptor(
                "laplace_double_layer",
                {"KERNEL_FUNCTION": "laplace_double_layer"},
                "default_dense",
            ),
            points,
            1,
            False,
            device_interface,
            precision,
            parameters=parameters,
        )
    )
