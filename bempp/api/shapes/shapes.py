"""Various built-in test shapes."""

import numpy as _np

def get_gmsh_file():
    """
    Create a new temporary gmsh file.

    Return a 3-tuple (geo_file,geo_name,msh_name), where
    geo_file is a file descriptor to an empty .geo file, geo_name is
    the corresponding filename and msh_name is the name of the
    Gmsh .msh file that will be generated.

    """
    import os
    import tempfile
    import bempp.api

    geo, geo_name = tempfile.mkstemp(
        suffix='.geo', dir=bempp.api.TMP_PATH, text=True)
    geo_file = os.fdopen(geo, "w")
    msh_name = os.path.splitext(geo_name)[0] + ".msh"
    return (geo_file, geo_name, msh_name)


def __generate_grid_from_gmsh_string(gmsh_string):
    """Return a grid from a string containing a gmsh mesh"""
    import os
    import tempfile

    if bempp.api.mpi_rank == 0:
        # First create the grid.
        handle, fname = tempfile.mkstemp(
            suffix='.msh', dir=bempp.api.TMP_PATH, text=True)
        with os.fdopen(handle, "w") as f:
            f.write(gmsh_string)
    grid = bempp.api.import_grid(fname)
    bempp.api.mpi_comm.Barrier()
    if bempp.api.mpi_rank == 0:
        os.remove(fname)
    return grid


def __generate_grid_from_geo_string(geo_string):
    """Helper routine that implements the grid generation
    """
    import os
    import subprocess
    import bempp.api

    def msh_from_string(geo_string):
        """Create a mesh from a string."""
        gmsh_command = bempp.api.GMSH_PATH
        if gmsh_command is None:
            raise RuntimeError("Gmsh is not found. Cannot generate mesh")
        f, geo_name, msh_name = get_gmsh_file()
        f.write(geo_string)
        f.close()

        fnull = open(os.devnull, 'w')
        cmd = gmsh_command + " -2 " + geo_name
        try:
            subprocess.check_call(
                cmd, shell=True, stdout=fnull, stderr=fnull)
        except:
            print("The following command failed: " + cmd)
            fnull.close()
            raise
        os.remove(geo_name)
        fnull.close()
        return msh_name

    msh_name = msh_from_string(geo_string)
    grid = bempp.api.import_grid(msh_name)
    os.remove(msh_name)
    return grid

def regular_sphere(refine_level):
    """
    Create a regular sphere with a given refinement level.

    Starting from an octahedron with 8 elements the grid is
    refined in each step by subdividing each element into
    four new elements, to create a sphere approximation.

    The number of elements in the final sphere is given as
    8 * 4**refine_level.

    The maximum allowed refinement level is 9.

    """
    from bempp.api.grid.grid import Grid
    import os
    import numpy as np

    if refine_level > 9:
        raise ValueError("'refine_level larger than 9 not supported.")

    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "regular_spheres.npz"
    )

    spheres = np.load(filename)
    return Grid(spheres["v" + str(refine_level)], spheres["e" + str(refine_level)])


def cube(length=1, origin=(0, 0, 0), h=0.1):
    """
    Return a cube mesh.

    Parameters
    ----------
    length : float
        Side length of the cube.
    origin : tuple
        Coordinates of the origin (bottom left corner)
    h : float
        Element size.

    """
    cube_stub = """
    Point(1) = {orig0,orig1,orig2,cl};
    Point(2) = {orig0+l,orig1,orig2,cl};
    Point(3) = {orig0+l,orig1+l,orig2,cl};
    Point(4) = {orig0,orig1+l,orig2,cl};
    Point(5) = {orig0,orig1,orig2+l,cl};
    Point(6) = {orig0+l,orig1,orig2+l,cl};
    Point(7) = {orig0+l,orig1+l,orig2+l,cl};
    Point(8) = {orig0,orig1+l,orig2+l,cl};

    Line(1) = {1,2};
    Line(2) = {2,3};
    Line(3) = {3,4};
    Line(4) = {4,1};
    Line(5) = {1,5};
    Line(6) = {2,6};
    Line(7) = {3,7};
    Line(8) = {4,8};
    Line(9) = {5,6};
    Line(10) = {6,7};
    Line(11) = {7,8};
    Line(12) = {8,5};

    Line Loop(1) = {-1,-4,-3,-2};
    Line Loop(2) = {1,6,-9,-5};
    Line Loop(3) = {2,7,-10,-6};
    Line Loop(4) = {3,8,-11,-7};
    Line Loop(5) = {4,5,-12,-8};
    Line Loop(6) = {9,10,11,12};

    Plane Surface(1) = {1};
    Plane Surface(2) = {2};
    Plane Surface(3) = {3};
    Plane Surface(4) = {4};
    Plane Surface(5) = {5};
    Plane Surface(6) = {6};

    Physical Surface(1) = {1};
    Physical Surface(2) = {2};
    Physical Surface(3) = {3};
    Physical Surface(4) = {4};
    Physical Surface(5) = {5};
    Physical Surface(6) = {6};

    Surface Loop (1) = {1,2,3,4,5,6};

    Volume (1) = {1};

    Mesh.Algorithm = 6;
    """

    cube_geometry = (
        "l = " + str(length) + ";\n" +
        "orig0 = " + str(origin[0]) + ";\n" +
        "orig1 = " + str(origin[1]) + ";\n" +
        "orig2 = " + str(origin[2]) + ";\n" +
        "cl = " + str(h) + ";\n" + cube_stub)

    return __generate_grid_from_geo_string(cube_geometry)
