def test_laplace_sphere():
    import bempp.api
    grid = bempp.api.shapes.sphere(h=0.1)
    space = bempp.api.function_space(grid, "DP", 0)
    slp = bempp.api.operators.boundary.laplace.single_layer(space, space, space)

    @bempp.api.real_callable
    def f(x, n, domain_index, result):
        result[0] = x[0] + 1

    rhs = bempp.api.GridFunction(space, fun=f)

    sol, info = bempp.api.linalg.gmres(slp, rhs)
