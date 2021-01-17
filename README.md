# Bempp-cl
Bempp-cl is an open-source boundary element method library that can be used to assemble all the standard integral kernels for
Laplace, Helmholtz, modified Helmholtz, and Maxwell problems. The library has a user-friendly Python interface that allows the
user to use BEM to solve a variety of problems, including problems in electrostatics, acoustics and electromagnetics.

Bempp-cl began life as BEM++, and was a Python library with a C++ computational core. The ++ slowly changed into pp as
functionality gradually moved from C++ to Python with only a few core routines remaining in C++. Bempp-cl is the culmination
of efforts to fully move to Python. It is an almost complete rewrite of Bempp: the C++ core has been replaced by highly SIMD
optimised just-in-time compiled OpenCL kernels, or alternatively, by just-in-time compiled Numba routines, which are
automatically used on systems that do not provide OpenCL drivers. User visible functionality is strictly separated from the
implementation of computational routines, making it easy to add other discretisation technologies in the future (e.g. future
support for SYCL-based heterogeneous compute devices).

## Installation
Bempp-cl can be installed from this repository by running:
```bash
python setup.py install
```

Full installation instuctions, including installation of dependencies, can be found at
[bempp.com/installation.html](https://bempp.com/installation.html).

On Arch Linux bempp-cl can be installed using the [AUR package](https://aur.archlinux.org/packages/python-bempp-cl).

## Documentation
Full documentation of Bempp can be found at [bempp.com/documentation](https://bempp.com/documentation/index.html)
and in [the Bempp Handbook](https://bempp.com/handbook).

## Getting help
Errors in the library should be added to the [GitHub issue tracker](https://github.com/bempp/bempp-cl/issues).

Questions about the library and its use can be asked on the [Bempp Discourse](https://bempp.discourse.group).

## Licence
Bempp-cl is licensed under an MIT licence. Full text of the licence can be found [here](LICENSE.md).
