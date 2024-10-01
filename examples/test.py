import os
import pytest
import nbformat
import sys
from nbconvert.preprocessors import ExecutePreprocessor

# Get all the noteboooks in each folder
scripts = []
for dir in ["laplace", "helmholtz", "maxwell", "other"]:
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir)
    for i in os.listdir(path):
        if i.endswith(".py"):
            scripts.append((path, i))


@pytest.mark.parametrize(("path", "script"), scripts)
def test_example(path, script, has_dolfin, has_dolfinx, dolfin_books_only):
    # Examples in this list will be skipped as the problems are very large or require GPUs
    if script in ["reentrant_cube_capacity.py", "opencl_benchmark.py",
                  "dirichlet_weak_imposition.py"]:
        pytest.skip()

    if not has_dolfin and script.endswith("dolfin.py"):
        try:
            import dolfin
        except ImportError:
            pytest.skip()
    if not has_dolfinx and script.endswith("dolfinx.py"):
        try:
            import dolfinx
        except ImportError:
            pytest.skip()
    if dolfin_books_only and "dolfin" not in script:
        pytest.skip()

    assert os.system(f"{sys.executable} {os.path.join(path, script)}") == 0
