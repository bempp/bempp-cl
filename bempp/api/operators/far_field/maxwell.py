"""Maxwell far-field operators."""


def electric_field(
    space, points, wavenumber, parameters=None, device_interface=None, precision=None
):
    """Return a Maxwell electric far-field potential operator."""
    from bempp.core.dense_potential_assembler import DensePotentialAssembler
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.operators import _add_wavenumber
    from bempp.api.assembly.potential_operator import PotentialOperator

    options = {}
    _add_wavenumber(options, wavenumber)

    return PotentialOperator(
        DensePotentialAssembler(
            space,
            OperatorDescriptor(
                "maxwell_electric_far_field", options, "maxwell_electric_far_field"
            ),
            points,
            3,
            True,
            device_interface,
            precision,
            parameters=parameters,
        )
    )


def magnetic_field(
    space, points, wavenumber, parameters=None, device_interface=None, precision=None
):
    """Return a Maxwell magnetic far-field potential operator."""
    from bempp.core.dense_potential_assembler import DensePotentialAssembler
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.operators import _add_wavenumber
    from bempp.api.assembly.potential_operator import PotentialOperator

    options = {}
    _add_wavenumber(options, wavenumber)

    return PotentialOperator(
        DensePotentialAssembler(
            space,
            OperatorDescriptor(
                "maxwell_magnetic_far_field", options, "maxwell_magnetic_far_field"
            ),
            points,
            3,
            True,
            device_interface,
            precision,
            parameters=parameters,
        )
    )

# def magnetic_field(
    # space, points, wavenumber, parameters=None, device_interface=None, precision=None
# ):
    # """Return a Maxwell magnetic field potential operator."""
    # from bempp.core.dense_potential_assembler import DensePotentialAssembler
    # from bempp.api.operators import OperatorDescriptor
    # from bempp.api.operators import _add_wavenumber
    # from bempp.api.assembly.potential_operator import PotentialOperator

    # options = {}
    # options["KERNEL_FUNCTION"] = "helmholtz_gradient"
    # _add_wavenumber(options, wavenumber)

    # return PotentialOperator(
        # DensePotentialAssembler(
            # space,
            # OperatorDescriptor(
                # "maxwell_magnetic_field_potential", options, "maxwell_magnetic_field"
            # ),
            # points,
            # 3,
            # True,
            # device_interface,
            # precision,
            # parameters=parameters,
        # )
    # )
