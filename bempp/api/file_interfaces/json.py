"""Import and export to JSON"""


def export_data_sets(file_name, grid_data_sets, description=""):
    """
    Export a list of grid datasets.

    Exports a list of grid datasets to JSON using the
    given file name.

    Parameters
    ----------
    file_name : string
        A given filename. Convention is that it should
        end in '.json'
    data_sets : list of GridDataSet objects
        A list of GridDataSet objects

    """
    from bempp.api.file_interfaces.general_interface import timestamp
    import json
    import codecs

    output = {
        "timestamp": timestamp(),
        "description": description,
        "grid_data_sets": [dataset.as_dict() for dataset in grid_data_sets],
    }

    with open(file_name, "wb") as outfile:
        json.dump(
            output,
            codecs.getwriter("utf-8")(outfile),
            sort_keys=True,
            indent=4,
            ensure_ascii=False,
        )


def import_data_sets(file_name=None, data_string=None):
    """
    Import data sets from a JSON string or file.

    Parameters
    ----------
    file_name : string
        A JSON file from which to load the data.
    data_string : string
        A string that contains the JSON data.

    Remarks
    -------
    Exactly one of the two parameters 'file_name' or 'data_string'
    must be provided.

    """
    import json
    from bempp.api.file_interfaces.general_interface import GridDataSet

    if file_name is None and data_string is None:
        raise ValueError(
            "Exactly one of 'file_name' or 'data_string' must be provided."
        )

    if "file_name" is not None:
        with open(file_name, "r") as infile:
            data_dict = json.load(infile)
    else:
        data_dict = json.loads(data_string)

    data_dict["grid_data_sets"] = [
        GridDataSet.from_dict(dataset) for dataset in data_dict["grid_data_sets"]
    ]

    return data_dict


def export(file_name, **kwargs):
    """
    Simple exporter for a grid or grid function.

    Parameters
    ----------
    file_name : string
        Filename to use.
    grid : Bempp Grid object
        A Bempp grid object to export
    grid_function : Bempp GridFunction object
        A Bempp grid function to export
    description : string
        A description of the GridDataSet object
    mode : string
        One of 'vertex' or 'element'. Describes
        wheter vertex values or element center values
        are stored (default 'vertex').
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

    Exactly one of 'grid' or 'grid_function' is allowed as keyword argument

    """
    import bempp.api
    from bempp.api.file_interfaces.general_interface import (
        bempp_object_to_grid_data_set,
    )

    if sum(["grid" in kwargs, "grid_function" in kwargs]) != 1:
        raise ValueError(
            "Exactly one of 'grid' or 'grid_function' must be provided." ""
        )

    if "transformation" in kwargs:
        mode = kwargs["transformation"]
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

    if "mode" in kwargs:
        mode = kwargs["mode"]
    else:
        mode = "vertex"

    if "grid" in kwargs:
        dataset = bempp_object_to_grid_data_set(
            kwargs["grid"],
            vertex_ids=vertex_ids,
            element_ids=element_ids,
            description=description,
        )

        export_data_sets(file_name, [dataset])

    if "grid_function" in kwargs:
        fun = kwargs["grid_function"]
        if mode == "vertex":
            dataset = bempp_object_to_grid_data_set(
                fun.space.grid,
                vertex_funs=[fun],
                vertex_ids=vertex_ids,
                element_ids=element_ids,
                description=description,
                transformation=transform,
            )
            export_data_sets(file_name, [dataset])
        if mode == "element":
            dataset = bempp_object_to_grid_data_set(
                fun.space.grid,
                element_funs=[fun],
                vertex_ids=vertex_ids,
                element_ids=element_ids,
                description=description,
                transformation=transform,
            )
            export_data_sets(file_name, [dataset])
