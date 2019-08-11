"""Routines for import and export."""
import meshio as _meshio
import numpy as _np


def import_grid(filename):
    """
    Import a grid.

    This routine uses the meshio library to export grids.
    A number of types are supported, including vtk, vtu,
    gmsh, dolphin xml. For a full list see

    https://github.com/nschloe/meshio

    """
    from bempp.api.grid.grid import Grid

    mesh = _meshio.read(filename)

    vertices = mesh.points.T
    elements = mesh.cells["triangle"].T.astype("uint32")

    try:
        domain_indices = mesh.cell_data["triangle"]["gmsh:physical"]
    except:
        domain_indices = None

    return Grid(vertices, elements, domain_indices=domain_indices)


def export(
    filename,
    grid=None,
    grid_function=None,
    data_type="node",
    transformation=None,
    write_binary=True,
):
    """
    Exporter for grids and grid functions.

    This method internally uses the meshio library. For a full
    list of supported data types see

    https://github.com/nschloe/meshio

    Note that export of domain indices is only possible for Gmsh (.msh)
    format files.

    Parameters
    ----------
    filename : string
        The name of the file to write out. The data type is
        chosen based on the file ending.
    grid : Grid object
        A grid object to export.
    grid_function : GridFunction object
        Grid function to export
    data_type : string
        Either 'node' for vertex data or 'element' for data
        at element centers.
    transformation : string or callable
        One of 'real', 'imag', 'abs', 'log_abs',
        None or a callable object. Transforms the
        data on input. A callable must return numpy
        arrays with the same number of dimensions as
        the input. If transformation is None the data
        is not modified.
    write_binary : Boolean
        Use binary format (write_binary=True) for the
        data if supported by the file format.
    """
    import os

    _, extension = os.path.splitext(filename)

    file_format = None

    if extension == ".msh":
        # Ensure that we use Gmsh2 for output
        # to preserve domain indices.
        # meshio does not yet support domain
        # indices for gmsh4.
        gmsh = True
        if write_binary:
            file_format = "gmsh2-binary"
        else:
            file_format = "gmsh2-ascii"
    else:
        gmsh = False

    if grid is not None and grid_function is not None:
        raise ValueError("Exactly one of 'grid' and 'grid_function' must be supplied.")

    cell_data = {"triangle": {}}
    point_data = None

    if grid_function is not None:
        grid = grid_function.space.grid

        if data_type == "node":
            data = _transform_array(
                grid_function.evaluate_on_vertices(), transformation
            ).T
            if _np.iscomplexobj(data):
                point_data = {"real": _np.real(data), "imag": _np.imag(data)}
            else:
                point_data = {"data": data}
        elif data_type == "element":
            data = _transform_array(
                grid_function.evaluate_on_element_centers(), transformation
            ).T
            if _np.iscomplexobj(data):
                cell_data["triangle"]["real"] = _np.real(data)
                cell_data["triangle"]["imag"] = _np.imag(data)
            else:
                cell_data["triangle"]["data"] = data

        else:
            raise ValueError("'data_type' must be one of 'element' or 'node'")

    cells = {"triangle": grid.elements.T.astype("int32")}
    points = grid.vertices.T

    if gmsh:
        cell_data["triangle"]["gmsh:physical"] = grid.domain_indices.astype("int32")
        unique_dom_indices = set(grid.domain_indices)
        unique_geom_indices = range(1, 1 + len(unique_dom_indices))
        geom_indices_map = dict(zip(unique_dom_indices, unique_geom_indices))
        geom_indices = _np.array([geom_indices_map[dom_index] for dom_index in grid.domain_indices], dtype='int32')    
        cell_data["triangle"]["gmsh:geometrical"] = geom_indices
    else:
        cell_data["triangle"]["domain index"] = grid.domain_indices.astype("int32")

    _meshio.write_points_cells(
        filename,
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        write_binary=write_binary,
        file_format=file_format,
    )


def _transform_array(a, mode=None):
    """
    Transform a data array.
    Parameters
    ----------
    a : np.ndarray
        Either a scalar array or a two dimensional data array.
    mode : string, callable or None
        One of 'real', 'imag', 'abs', 'log_abs', 'abs_squared',
        a transformation callable or None. The callable needs to take
        an input array of dimension 2 and return an array of
        dimension 2. If mode is None the input array is not modified.
    """
    if mode is None:
        return a

    ndim = a.ndim
    if ndim == 1:
        _np.expand_dims(a, 1)

    if mode == "real":
        res = _np.real(a)
    elif mode == "imag":
        res = _np.imag(a)
    elif mode == "abs":
        res = _np.sqrt(_np.sum(_np.abs(a) ** 2, axis=0, keepdims=True))
    elif mode == "abs_squared":
        res = _np.sum(_np.abs(a) ** 2, axis=0, keepdims=True)
    elif mode == "log_abs":
        res = _np.log(_np.sqrt(_np.sum(_np.abs(a) ** 2, axis=0, keepdims=True)))
    else:
        res = mode(a)

    if ndim == 1:
        return _np.squeeze(res, axis=0)
    else:
        return res
