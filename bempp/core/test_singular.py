import bempp.api

bempp.api.enable_console_logging('timing')

grid = bempp.api.shapes.regular_sphere(3)
space = bempp.api.function_space(grid, "P", 1)
op = bempp.api.operators.boundary.laplace.single_layer(space, space, space, assembler="only_singular_part")
mat = op.weak_form().A
