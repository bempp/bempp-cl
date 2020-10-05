---
title: 'Bempp-cl: A fast Python based just-in-time compiling boundary element library.'
tags:
  - Python
  - OpenCL
  - boundary element method
  - partial differential equations
  - numerical analysis
authors:
  - name: Timo Betcke
    orcid: 0000-0002-3323-2110
    affiliation: 1
  - name: Matthew W. Scroggs
    orcid: 0000-0002-4658-2443
    affiliation: 2
affiliations:
 - name: Department of Mathematics, University College London
   index: 1
 - name: Department of Engineering, University of Cambridge
   index: 2
date: 15 January 2020
bibliography: paper.bib
---

# Summary

The boundary element method (BEM) is a numerical method for approximating the solution of certain types of partial 
differential equations (PDEs) in homogeneous bounded or unbounded domains. The method finds the approximation by discretising 
a boundary integral equation that can be derived from the PDE. The mathematical background of BEM is covered in, for example, 
@Stein07 or @McLean. Typical applications of BEM include electrostatic problems, and acoustic and electromagnetic scattering.

Bempp-cl is an open-source boundary element method library that can be used to assemble all the standard integral kernels for
Laplace, Helmholtz, modified Helmholtz, and Maxwell problems. The library has a user-friendly Python interface that allows the
user to use BEM to solve a variety of problems, including problems in electrostatics, acoustics and electromagnetics.

Bempp-cl began life as BEM++, and was a Python library with a C++ computational core. The ++ slowly changed into pp as 
functionality gradually moved from C++ to Python with only a few core routines remaining in C++. Bempp-cl is the culmination 
of efforts to fully move to Python, and is an almost complete rewrite of Bempp/

In this paper, we give an overview of the functionality of Bempp-cl and present highlights of the library's recent developments.
An overview of the original version of the library is presented in @bemppold. Full documentation of the library can be found
online at ``bempp.com`` and in @bempphandbook.

# Statement of need



# An overview of Bempp features

Bempp-cl is divided into two parts: `bempp.api` and `bempp.core`.
The user interface of the library is contained in `bempp.api`.
The core assembly routines of the library are contained in `bempp.core`. The majority of users of Bempp-cl are unlikely to need
to directly interact with the functionality in `bempp.core`.

There are five main steps that are commonly taken when solving a problem with BEM:

1. First a surface grid (or mesh) must be created on which to solve the problem.
2. Finite dimensional function spaces are defined on this grid.
3. Boundary integral operators acting on the function spaces are defined.
4. The operators are discretised and the resulting linear systems solved.
5. Domain potentials and far field operators can be used to evaluate the solution away from the boundary.

Bempp-cl provides routines that implement each of these steps.

## Grid Interface
The submodule `bempp.api.shapes` contains the definitions of a number of shapes. From these, grids with various element sizes 
can be created internally using Gmsh [@gmsh]. Alternatively, meshes can be imported from many formats using the meshio library 
[@meshio]. Bempp-cl currently only supports flat triangle based surface meshes. Higher-order triangular meshes may be 
supported in the future.

## Function Spaces
Bempp-cl provides piecewise constant and piecewise linear (continuous and discontinuous) function spaces for solving scalar 
problems. For Maxwell problems, Bempp-cl can create Rao--Wilton--Glisson [@rwg] div-conforming spaces and Nédélec 
[@nedelec] curl-conforming spaces. In addition to these, Bempp-cl can also generate constant and linear spaces on the 
barycentric dual grid as well as Buffa--Christiansen div-conforming spaces, as described in @bc. These spaces can all be 
created using the `bempp.api.function_space` command.

## Boundary operators
Boundary operators for Laplace, Helmholtz, modified Helmholtz and Maxwell problems can be found in the 
`bempp.api.operators.boundary` submodule, as well as sparse identity operators. For Laplace and Helmholtz problems, Bempp-cl 
can create single layer, double layer, adjoint double layer and hypersingular operators. For Maxwell problems, both electric 
field and magnetic field operators can be used. For formulations involving the product of operators, such as Calder\'on 
preconditioned Maxwell problems [@maxwellbempp], Bempp-cl contains an operator algebra that allows the product to be easily 
obtained. This operator algebra is described in detail in @operatoralg.

Internally, Bempp-cl uses PyOpenCL [@pyopencl] to just-in-time compile its operator assembly routines on a wide range of CPU
and GPU compute devices. On systems without OpenCL support, Numba [@numba] is used to just-in-time compile
Python-based assembly kernels, giving a slower but still viable alternative to OpenCL.

Bempp-cl provides an interface to the Exafmm-t library [@exafmm] for faster assembly of larger problems with lower memory
requirements using the fast multipole method (FMM). The interface to Exafmm-t is writting in a generic way so that other
FMM libraries or alternative matrix compression techniques could be used in future. 

## Discretisation and solvers
Operators are assembled using OpenCL or Numba based dense assembly, or via interface to fast multipole methods (FMM, see next 
section). The submodule `bempp.api.linalg` contains wrapped versions of SciPy's [@scipy] LU, CG, and GMRes solvers. By using 
SciPy's `LinearOperator` interface, Bempp-cl's boundary operators can easily be used with other iterative solvers.

## Potential and far field operators
Potential and far fiels operators for the evaluation at points in the domain or the asymptotic behavior at infinity are 
included in the `bempp.api.operators.potential` and `bempp.api.operators.far_field` submodules.

## Further information
The [Bempp-cl repository](https://github.com/bempp/bempp-cl) contains a growing list of example Jupyter notebooks. Moreover, 
on the [Bempp hompepage](https://bempp.com) an in-development Bempp-cl handbook is hosted that contains details about the 
functionality of Bempp-cl.

# Acknowledgements
We would like to thank the Exafmm team [@exafmm], and here in particular Lorena Barba and Tingyu Wang for their efforts to 
integrate Exafmm-t into Bempp-cl. We further thank the HyENA team [@hyena] at Graz University of Technology who provided C++ 
definitions of core numerical quadrature rules, which were translated to Python as part of the development effort for 
Bempp-cl.
    
# References
