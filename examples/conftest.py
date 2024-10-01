import pytest


def pytest_addoption(parser):
    parser.addoption("--has-dolfin", type="int", default="0")
    parser.addoption("--has-dolfinx", type="int", default="0")
    parser.addoption("--has-exafmm", type="int", default="0")
    parser.addoption("--dolfin-books-only", type="int", default="0")


@pytest.fixture
def has_dolfin(request):
    return request.config.getoption("--has-dolfin") > 0


@pytest.fixture
def has_dolfinx(request):
    return request.config.getoption("--has-dolfinx") > 0


@pytest.fixture
def has_exafmm(request):
    return request.config.getoption("--has-exafmm") > 0


@pytest.fixture
def dolfin_books_only(request):
    return request.config.getoption("--dolfin-books-only") > 0
