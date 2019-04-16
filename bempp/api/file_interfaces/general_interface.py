"""The core module containing format independent routines."""
import numpy as _np

# Numpy data requirements
_np_require = ["A", "O", "C"]


def bempp_grid_from_data_set(grid_data_set):
    """Convert a grid data set to Bempp grid."""
    from bempp.api.grid import Grid

    return Grid(
        grid_data_set.grid.vertices,
        grid_data_set.grid.elements,
        grid_data_set.grid.domain_indices,
    )


def export(file_name, **kwargs):
    """
    Simple exporter for a grid or grid function.

    This funtion can export grids and gridfunctions into external file formats.
    Supported formats are:

    * Gmsh ASCII v2.2 files
    * JSON Dictonaries
    * VTK Legacy ASCII files

    The function only takes keyword arguments.

    Parameters
    ----------
    file_name : string
        Filename to use.
    grid : Bempp Grid object
        A Bempp grid object to export
    grid_function : Bempp GridFunction object
        A Bempp grid function to export
    data_series : A list of list of grid functions
        A two dimensional list of grid functions, 
        where the entry (i, j) is the jth time data
        in the ith data series.
    timesteps : If a data series is given this is
        an optional array of time values associated
        with time series objects. If None is given the
        steps [0, 1, ...] are assumed.
    description : string
        A description of the GridDataSet object
    data_type : string
        One of 'node' or 'element'. Describes
        wheter vertex values or element center values
        are stored (default 'node').
    transformation : string or function object
        One of 'real', 'imag', 'abs', 'log_abs',
        None or a callable object. Transforms the
        data on input. A callable must return numpy
        arrays with the same number of dimensions as
        the input. If transformation is None the data
        is not modified.
    vertex_ids : np.ndarray
        An optional uint32 array of vertex ids
    element_ids : np.ndarray
        An optional uint32 array of element ids

    Exactly one of 'grid', 'grid_function' or data series 
    is allowed as keyword argument

    """
    import bempp.api
    import os

    export_fun = None

    _, extension = os.path.splitext(file_name)

    if extension == ".msh":
        # Format is Gmsh
        from bempp.api.file_interfaces.gmsh import export_data_sets

        export_fun = export_data_sets
    elif extension == ".json":
        # Format is JSON
        from bempp.api.file_interfaces.json import export_data_sets

        export_fun = export_data_sets
    elif extension == ".vtk":
        # Format is VTK
        from bempp.api.file_interfaces.vtk import export_data_sets

        export_fun = export_data_sets
    else:
        raise ValueError("Unsupported file format.")

    if sum(["grid" in kwargs, "grid_function" in kwargs, "data_series" in kwargs]) != 1:
        raise ValueError(
            "Exactly one of 'grid', 'grid_function' or 'data_series'  must be provided."
            ""
        )

    if "transformation" in kwargs:
        transform = kwargs["transformation"]
    else:
        transform = None

    if "description" in kwargs:
        description = kwargs["description"]
    else:
        description = ""

    if "vertex_ids" in kwargs:
        vertex_ids = kwargs["vertex_ids"]
    else:
        vertex_ids = None

    if "element_ids" in kwargs:
        element_ids = kwargs["element_ids"]
    else:
        element_ids = None

    if "data_type" in kwargs:
        mode = kwargs["data_type"]
    else:
        mode = "node"

    if "timesteps" in kwargs:
        timesteps = kwargs["timesteps"]
    else:
        timesteps = None

    funs = None

    if "grid" in kwargs:
        dataset = bempp_object_to_grid_data_set(
            kwargs["grid"],
            vertex_ids=vertex_ids,
            element_ids=element_ids,
            description=description,
        )

        export_fun(file_name, [dataset])
        return

    if "data_series" in kwargs:
        funs = kwargs["data_series"]

    if "grid_function" in kwargs:
        funs = [[kwargs["grid_function"]]]

    if mode == "node":
        dataset = bempp_object_to_grid_data_set(
            funs[0][0].space.grid,
            vertex_funs=funs,
            vertex_ids=vertex_ids,
            element_ids=element_ids,
            description=description,
            transformation=transform,
            timesteps=timesteps,
        )
        export_fun(file_name, [dataset])
    if mode == "element":
        dataset = bempp_object_to_grid_data_set(
            funs[0][0].space.grid,
            element_funs=funs,
            vertex_ids=vertex_ids,
            element_ids=element_ids,
            description=description,
            transformation=transform,
            timesteps=timesteps,
        )
        export_fun(file_name, [dataset])


def import_grid(file_name, return_id_maps=False):
    """
    Import grids into Bempp

    The file ending is used by Bempp to recognize the
    format. Currently supported are:

    .msh    : Gmsh Version 2.2 ASCII or Binary files
    .json   : A JSON file that stores data arrays
              using a base64 encoding.

    Parameters
    ----------
    file_name : string
        Name of the file from which to read the grid.
    return_id_maps : bool
        If True return a triple consisting of
        (grid, vertex_ids, element_ids), where the vertex_ids
        and element_ids are maps such that for element i
        the associated file index is element_ids[i], and
        similar for the vertex ids. This helps to translate
        date between files and Bempp's internal format.

    Returns
    -------
    grid : bempp.api.Grid
        A grid object

    Examples
    --------
    To read a Gmsh grid file called 'grid_file.msh' use

    >>> grid = import_grid('grid_file.msh')

    To also obtain the vertex and element ids use

    >>> grid, vertex_ids, element_ids = import_grid(
            'grid_file.msh', return_id_maps=True)

    """
    import os

    _, extension = os.path.splitext(file_name)

    data_sets = None

    if extension == ".msh":
        # Format is Gmsh
        from bempp.api.file_interfaces.gmsh import import_data_sets

        data_sets = import_data_sets(file_name)
    elif extension == ".json":
        # Format is JSON
        from bempp.api.file_interfaces.json import import_data_sets

        data_sets = import_data_sets(file_name)
    else:
        raise ValueError("Unsupported file format.")

    grid = bempp_grid_from_data_set(data_sets["grid_data_sets"][0])

    if return_id_maps:
        return (
            grid,
            data_sets["grid_data_sets"][0].grid.vertex_ids,
            data_sets["grid_data_sets"][0].grid.element_ids,
        )
    else:
        return grid


class GenericGrid(object):
    """
    A simple object describing a grid.

    Attributes
    ----------
    vertices : np.ndarray
        A 3 x N Numpy array of N vertices describing the grid.
        The type of this array is 'float64'
    elements : np.ndarray
        A 3 x M Numpy array of M elements. The type of this array
        is 'uint32'
    vertex_ids : np.ndarray
        A uint32 array of vertex ids. By default vertex_ids is
        identical to range(0, N). Useful if vertices should
        be stored under specific ids within a given file format.
        The vertex id of vertex j is given by vertex_ids[j]
    element_ids : np.ndarray
        A uint32 array of element ids. By default element_ids
        is identical to range(0, M). Useful if elements
        should be stored under specific ids withing a given file
        format. The element id of element j is given by
        element_ids[j]
    description : string
        A description or identifier of the grid.
    domain_indices : np.ndarray
        A uint32 array of domain indices. These are used to 
        group elements into physical regions.
    number_of_vertices : int
        The number of vertices
    number_of_elements : int
        The number of elements

    Remarks
    -------
    All Numpy arrays will be stored in C (row-major) order.

    No sanity check will be performed on the input data. It is assumed that
    it describes valid grid data.

    """

    def __init__(
        self,
        vertices,
        elements,
        vertex_ids=None,
        element_ids=None,
        description="",
        domain_indices=None,
    ):
        """
        Construct a grid object.

        Parameters
        ----------
        vertices : np.ndarray
            A 3 x N Numpy array of N vertices describing the grid.
            The type of this array is 'float64'
        elements : np.ndarray
            A 3 x M Numpy array of M elements. The type of this array
            is 'uint32'
        vertex_ids : np.ndarray
            A uint32 array of vertex ids. By default vertex_ids is
            identical to range(0, N). Useful if vertices should
            be stored under specific ids within a given file format.
            The vertex id of vertex j is given by vertex_ids[j]
        element_ids : np.ndarray
            A uint32 array of element ids. By default element_ids
            is identical to range(0, M). Useful if elements
            should be stored under specific ids withing a given file
            format. The element id of element j is given by
            element_ids[j]
        description : string
            A description or identifier of the grid.
        domain_indices : np.ndarray
            An uint32 array of domain indices for the elements. If 
            domain_indices is None each element is associated 
            the default index 0.

        Remarks
        -------
        If input arrays are not in the specified formats a conversion
        of the data is attempted.

        """
        self._vertices = _np.require(vertices, "float64", _np_require)
        self._elements = _np.require(elements, "uint32", _np_require)

        if self._vertices.shape[0] != 3:
            raise ValueError("Axis 0 of 'vertices' must have length 3.")

        if self._elements.shape[0] != 3:
            raise ValueError("Axis 0 of 'elements' must have length 3.")

        self._number_of_vertices = self._vertices.shape[1]
        self._number_of_elements = self._elements.shape[1]

        if vertex_ids is None:
            self._vertex_ids = _np.arange(self._number_of_vertices, dtype="uint32")
        else:
            if len(vertex_ids) != self._number_of_vertices:
                raise ValueError("Length of vertex_ids != number of vertices")
            self._vertex_ids = _np.require(vertex_ids, "uint32", _np_require)

        if element_ids is None:
            self._element_ids = _np.arange(self._number_of_elements, dtype="uint32")
        else:
            self._element_ids = _np.require(element_ids, "uint32", _np_require)

        self._description = description

        if domain_indices is None:
            self._domain_indices = _np.zeros(self._number_of_elements, dtype="uint32")
        else:
            if len(domain_indices) != self._number_of_elements:
                raise ValueError(
                    "Length of 'domain_indices' must be equal to the number of elements."
                )
            self._domain_indices = _np.require(domain_indices, "uint32", _np_require)

    @property
    def vertices(self):
        """Return vertices."""
        return self._vertices

    @property
    def elements(self):
        """Return elements."""
        return self._elements

    @property
    def description(self):
        """Return description"""
        return self._description

    @property
    def number_of_vertices(self):
        """Return number of vertices"""
        return self._number_of_vertices

    @property
    def number_of_elements(self):
        """Return number of elements."""
        return self._number_of_elements

    @property
    def vertex_ids(self):
        """Return vertex ids"""
        return self._vertex_ids

    @property
    def element_ids(self):
        """Return element ids"""
        return self._element_ids

    @property
    def domain_indices(self):
        """Return domain indices."""
        return self._domain_indices

    def as_dict(self):
        """
        Return a serializable dictionary with the class data.
        
        All Numpy arrays are converted to base64 encoded strings. 
        """
        return {
            "vertices": numpy_to_base64(self.vertices),
            "elements": numpy_to_base64(self.elements),
            "number_of_vertices": self.number_of_vertices,
            "number_of_elements": self.number_of_elements,
            "vertex_ids": numpy_to_base64(self.vertex_ids),
            "element_ids": numpy_to_base64(self.element_ids),
            "description": self.description,
            "domain_indices": numpy_to_base64(self.domain_indices),
        }

    @classmethod
    def from_dict(cls, d):
        """Recover the object from a dictionary of serialized data fields."""
        return GenericGrid(
            base64_to_numpy(d["vertices"], "float64", (3, d["number_of_vertices"])),
            base64_to_numpy(d["elements"], "uint32", (3, d["number_of_elements"])),
            base64_to_numpy(d["vertex_ids"], "uint32", (d["number_of_vertices"],)),
            base64_to_numpy(d["element_ids"], "uint32", (d["number_of_elements"],)),
            d["description"],
            base64_to_numpy(d["domain_indices"], "uint32", (d["number_of_elements"],)),
        )


class Data(object):
    """
    A set that describes a single or multiple data arrays.

    A Data object describes set of assocated data arrays,
    such as data outputs resulting from a frequency sweep
    or different time steps.

    Attributes
    ----------
    real : list of arrays
        A list of nc x N arrays of type 'float64'. Here, nc
        denotes the number of components of each data point and
        N is identical to the number of data points. It stores the
        real part of the given data.
    imag : list of arrays or None
        if dtype == 'complex' a list of nc x N arrays of type 'float64'.
        It stores the imaginary part of the given data.
    description : string
        A string describing the data.
    number_of_arrays : integer
        The number of data arrays in 'data'
    number_of_components : integer
        The number of components in the data sets.
    number_of_data_points : integer
        The number of data points
    dtype: string
        Either 'float' for real double precision data or
        'complex' for complex double precision data.

    """

    def __init__(self, components, npoints, description="", data=None, dtype=None):
        """
        Initialize a data array.

        Parameters
        ----------
        components : integer
            The number of components in each data point
        npoints : integer
            The number of points in each stored data array
        data : list of ndarrays
            Optional list of data arrays of type 'float' 
            or 'complex' in C ordering and of dimension 
            (components x npoints).
        description : string
            A description of the data
        dtype: string
            Either 'float' for real double precision data or
            'complex' for complex double precision data. 'dtype'
            can be None if the 'data' parameter is provided. Then
            the type is determined from the input data.
        """
        self._components = components
        self._npoints = npoints
        self._description = description
        self._real = None
        self._imag = None

        if data is None:
            if dtype not in ["complex", "float"]:
                raise ValueError("'dtype' must be either real or 'complex'")
            else:
                self._dtype = dtype
                self._real = []
                if dtype == "complex":
                    self._imag = []
        else:
            iscomplex = _np.any([_np.iscomplexobj(a) for a in data])
            if iscomplex:
                self._dtype = "complex"
            else:
                self._dtype = "float"
            for a in data:
                if a.shape != (components, npoints):
                    raise ValueError(
                        "Wrong shape. {0} != {1}".format(a.shape, (components, npoints))
                    )
            if self._dtype == "float":
                self._real = [_np.require(a, "float64", _np_require) for a in data]
            else:
                self._real = [
                    _np.require(_np.real(a), "float64", _np_require) for a in data
                ]
                self._imag = [
                    _np.require(_np.imag(a), "float64", _np_require) for a in data
                ]

    def add_data(self, data_array):
        """
        Add a data array.

        The data array must have the dimension components x npoints
        and must be convertible to self.dtype

        """
        d = _np.require(data_array, self.dtype, _np_require)
        if d.shape != (self.components, self.number_of_data_points):
            raise ValueError("'data' has wrong dimensions.")
        if self.dtype == "float":
            self._real.append(_np.require(data_array, "float64", _np_require))
        else:
            self._real.append(_np.require(_np.real(data_array), "float64", _np_require))
            self._imag.append(_np.require(_np.imag(data_array), "float64", _np_require))

    @property
    def real(self):
        """Return real part"""
        return self._real

    @property
    def imag(self):
        """
        Return imaginary part.
        
        If self.dtype == 'float' this attribute returns None 

        """
        return self._imag

    @property
    def dtype(self):
        """Return data type."""
        return self._dtype

    @property
    def description(self):
        """Return description"""
        return self._description

    @property
    def components(self):
        """Return number of components"""
        return self._components

    @property
    def number_of_data_points(self):
        """Return the number of data points"""
        return self._npoints

    @property
    def number_of_arrays(self):
        """Return the number of arrays"""
        return len(self.real)

    def as_dict(self):
        """
        Return a serializable dictionary with the class data.
        
        All Numpy arrays are converted to base64 encoded strings. 
        """
        return {
            "real": [numpy_to_base64(a) for a in self.real],
            "imag": [numpy_to_base64(a) for a in self.imag]
            if self.imag is not None
            else None,
            "dtype": self.dtype,
            "description": self.description,
            "components": self.components,
            "number_of_data_points": self.number_of_data_points,
            "number_of_arrays": self.number_of_arrays,
        }

    @classmethod
    def from_dict(cls, d):
        """Recover the object from a dictionary of serialized data fields."""
        if d["dtype"] == "float":
            return Data(
                d["components"],
                d["number_of_data_points"],
                d["description"],
                [
                    base64_to_numpy(
                        a, "float64", (d["components"], d["number_of_data_points"])
                    )
                    for a in d["real"]
                ],
            )
        if d["dtype"] == "complex":
            return Data(
                d["components"],
                d["number_of_data_points"],
                d["description"],
                [
                    (
                        base64_to_numpy(
                            a, "float64", (d["components"], d["number_of_data_points"])
                        )
                        + 1j
                        * base64_to_numpy(
                            b, "float64", (d["components"], d["number_of_data_points"])
                        )
                    )
                    for a, b in zip(d["real"], d["imag"])
                ],
            )


class GridDataSet(object):
    """
    A complete dataset consisting of a grid and associated data
    
    Attributes
    ----------
    grid : GenericGrid
        Return the grid containing vertices and
        elements
    element_data : list of Data
        List of Data objects associated with the elements
    vertex_data : list of Data
        List of Data objects associated with the nodes
    description : string
        A description of the data
    timesteps : np.ndarray
        A 'float64' array of real time values associated with
        each of the data array. By default no timesteps are assumed.
        The number of timesteps must be identical to the number of data
        arrays in each data section.
    number_of_time_steps : uint32
        The number of timesteps in the dataset.

    """

    def __init__(
        self, grid, description="", element_data=None, vertex_data=None, timesteps=None
    ):
        """
        Initialize a GridDataSet object

        Paramters
        ---------
        grid : GenericGrid
            The Grid object describing the grid
        description : string
            A description string for the GridDataSet
        element_data : list of Data objects
            Various data objects associated with the elements
        vertex_data : list of Data objects
            Various data objects associated with the vertices
        timesteps : np.ndarray
            A 'float64' array of real time values associated with
            each of the data array. By default the values
            [0, 1, 2, ...] are associated. Can be modified to reflect
            data arrays that have been added.
            
        """
        self._grid = grid
        self._element_data = [] if element_data is None else element_data
        self._vertex_data = [] if vertex_data is None else vertex_data
        self._description = description
        self._timesteps = (
            _np.array([0], dtype="float64") if timesteps is None else timesteps
        )

    def add_element_data(self, data):
        """Add an element data set."""
        if data.number_of_data_points != grid.number_of_elements:
            raise ValueError("Number of data points not equal to number of elements.")
        self._element_data.append(data)

    def add_vertex_data(self, data):
        """Add a vertex data set."""
        if data.number_of_data_points != grid.number_of_vertices:
            raise ValueError("Number of data points not equal to number of vertices.")
        self._vertex_data.append(data)

    @property
    def grid(self):
        """Return the grid."""
        return self._grid

    @property
    def element_data(self):
        """Return the element data."""
        return self._element_data

    @property
    def vertex_data(self):
        """Return the vertex data."""
        return self._vertex_data

    @property
    def description(self):
        """Return the description of this grid dataset."""
        return self._description

    @property
    def timesteps(self):
        """Return the time steps."""
        return self._timesteps

    @timesteps.setter
    def timesteps(self, values):
        """Set the timestep values."""
        if len(values) != self.number_of_arrays:
            raise ValueError(
                "Number of arrays must be identical to number of time steps."
            )
        self._timesteps = _np.require(values, "float64", _np_require)

    @property
    def number_of_timesteps(self):
        """Return the number of timesteps."""
        return len(self.timesteps)

    def as_dict(self):
        """
        Return a serializable dictionary with the class data.
        
        All Numpy arrays are converted to base64 encoded strings. 
        """
        return {
            "grid": self.grid.as_dict(),
            "element_data": [data.as_dict() for data in self.element_data],
            "vertex_data": [data.as_dict() for data in self.vertex_data],
            "description": self.description,
            "timesteps": numpy_to_base64(self.timesteps),
            "number_of_timesteps": self.number_of_timesteps,
        }

    @classmethod
    def from_dict(cls, d):
        """Recover the object from a dictionary of serialized data fields."""
        return GridDataSet(
            GenericGrid.from_dict(d["grid"]),
            description=d["description"],
            vertex_data=[Data.from_dict(d) for d in d["vertex_data"]],
            element_data=[Data.from_dict(d) for d in d["element_data"]],
            timesteps=base64_to_numpy(
                d["timesteps"], "float64", (d["number_of_timesteps"],)
            ),
        )


def numpy_to_base64(data):
    """
    Convert binary array data to a string.

    Convert a numpy array to a UTF-8 string using base64
    encoding and decoding to UTF-8.

    Data is assumed to be C contiguous.
    """
    import base64

    if not data.flags["C_CONTIGUOUS"]:
        raise ValueError("data must be contiguous in memory.")
    return base64.b64encode(data).decode("utf-8")


def base64_to_numpy(data, dtype, dims):
    """
    Convert string to array data

    Convert a base64 UTF-8 decoded string to a numpy array  
    given type and dimensions.
    """
    import base64

    a = _np.frombuffer(base64.decodebytes(data.encode("utf-8")), dtype=dtype)
    return _np.asfortranarray(a.reshape(dims))


def transform_array(a, mode=None):
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


def bempp_object_to_grid_data_set(bempp_grid, **kwargs):
    """

    Parameters
    ----------
    bempp_grid : A Bempp grid object
    vertex_funs : list of list of Bempp GridFunction objects
        This is a two-dimensional list of grid function objects.
        The grid function (i, j) is represented as the function
        at the jth time-point for the ith data series.
        The grid functions are evaluated at the nodes
        and the nodal data is stored. 
    element_funs : list of list of Bempp GridFunction objects
        This is a two-dimensional list of grid function objects.
        The grid function (i, j) is represented as the function
        at the jth time-point for the ith data series.
        The grid functions are evaluated at the element
        centers and this data stored.
    timesteps : If a data series is given this is
        an optional array of time values associated
        with time series objects. If None is given the
        steps [0, 1, ...] are assumed.
    transformation : string or function object
        One of 'real', 'imag', 'abs', 'log_abs',
        None or a callable object. Transforms the
        data on input. A callable must return numpy
        arrays with the same number of dimensions as
        the input. If transformation is None the data
        is not modified.
    description : string
        A description of the GridDataSet object
    vertex_ids : np.ndarray
        An optional uint32 array of vertex ids
    element_ids : np.ndarray
        An optional uint32 array of element ids

    Remarks
    -------
    All given input data must be associated with the same grid.

    """
    import bempp.api

    if "transformation" in kwargs:
        mode = kwargs["transformation"]
    else:
        mode = None

    if "description" in kwargs:
        description = kwargs["description"]
    else:
        description = ""

    if "vertex_ids" in kwargs:
        vertex_ids = kwargs["vertex_ids"]
    else:
        vertex_ids = None

    if "element_ids" in kwargs:
        element_ids = kwargs["element_ids"]
    else:
        element_ids = None

    if "timesteps" in kwargs:
        timesteps = kwargs["timesteps"]
    else:
        timesteps = None

    grid = GenericGrid(
        bempp_grid.vertices,
        bempp_grid.elements,
        vertex_ids,
        element_ids,
        description="",
    )

    vertex_data = None
    element_data = None

    if "vertex_funs" in kwargs:
        vertex_data = []
        for data_series in kwargs["vertex_funs"]:
            components = data_series[0].component_count
            npoints = bempp_grid.leaf_view.entity_count(2)
            if data_series[0].dtype == "float":
                dtype = "float"
            else:
                dtype = "complex"
            data_container = Data(components, npoints, dtype=dtype)
            vertex_data.append(data_container)
            for fun in data_series:
                if fun.space.grid != bempp_grid:
                    raise ValueError("Grids do not agree.")
                data_container.add_data(
                    transform_array(fun.evaluate_on_vertices(), mode)
                )
    if "element_funs" in kwargs:
        element_data = []
        for data_series in kwargs["element_funs"]:
            components = data_series[0].component_count
            npoints = bempp_grid.leaf_view.entity_count(0)
            if data_series[0].dtype == "float":
                dtype = "float"
            else:
                dtype = "complex"
            data_container = Data(components, npoints, dtype=dtype)
            element_data.append(data_container)
            for fun in data_series:
                if fun.space.grid != bempp_grid:
                    raise ValueError("Grids do not agree.")
                data_container.add_data(
                    transform_array(fun.evaluate_on_element_centers(), mode)
                )
    return GridDataSet(
        grid, description, element_data, vertex_data, timesteps=timesteps
    )


def timestamp():
    """Return a current time stamp."""
    import datetime

    return "{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now())
