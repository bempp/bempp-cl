"""Import and export routines for Gmsh."""

# The following is needed for the binary
# representation of the Gmsh datatype.
_nodes_per_elem_type = {
    1: 2,
    2: 3,
    3: 4,
    4: 4,
    5: 8,
    6: 6,
    7: 5,
    8: 3,
    9: 6,
    10: 9,
    11: 10,
    12: 27,
    13: 18,
    14: 14,
    15: 1,
    16: 8,
    17: 20,
    18: 15,
    19: 13,
    20: 9,
    21: 10,
    22: 12,
    23: 15,
    24: 15,
    25: 21,
    26: 4,
    27: 5,
    28: 6,
    29: 20,
    30: 35,
    31: 56,
    92: 64,
    93: 125,
}


def parse_gmsh(file_name):
    """Import an ascii based Gmsh file."""
    from bempp.api.file_interfaces.general_interface import GenericGrid
    import numpy as np
    import sys

    def read_version(s):
        """Read the Gmsh file version."""
        tokens = s.split()
        if len(tokens) != 3:
            raise ValueError("File header has unsupported format.")
        try:
            version = float(tokens[0])
        except:
            raise ValueError("Version number not recognized.")
        return version

    def read_vertex(s):
        """Read a vertex."""
        tokens = s.split()
        if len(tokens) != 4:
            raise ValueError("Unsupported format for vertex in string {0}".format(s))
        try:
            index = int(tokens[0])
            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])
        except:
            raise ValueError("Vertex value not recognized in string %s", s)

        return index, x, y, z

    def read_element(s):
        """Read an element."""
        tokens = s.split()
        try:
            index = int(tokens[0])
            elem_type = int(tokens[1])
        except:
            raise ValueError("Unspported format for element in string %s", s)
        if elem_type != 2:
            return None
        try:
            phys_id = int(tokens[3])
            v2 = int(tokens[-1])
            v1 = int(tokens[-2])
            v0 = int(tokens[-3])
        except:
            raise ValueError("Unsupported format for element in string {0}".format(s))
        return index, v0, v1, v2, phys_id

    all_vertices = None
    all_vertex_ids = None
    elements_list = None
    elements_ids_list = None
    domain_indices_list = None

    binary = False
    data_size = None
    from IPython import embed

    with open(file_name, "rb") as f:
        while True:
            s = f.readline().rstrip().decode("utf-8")
            if s == "":
                break
            if s == "$MeshFormat":
                s = f.readline().rstrip().decode("utf-8")
                tokens = s.split()
                if tokens[0] != "2.2":
                    raise ValueError("Only Gmsh 2.2 file format supported.")
                if int(tokens[1]) == 1:
                    binary = True
                data_size = int(tokens[2])
                if binary:
                    # Read endian line but do not process it.
                    # Always assume little endian (Intel type machines)
                    import struct

                    val = struct.unpack("i", f.read(4))[0]
                    if val != 1:
                        raise ValueError("Error in byte ordering. Try read as ascii.")
                    f.readline()
                # pylint: disable=protected-access
                s = f.readline().rstrip().decode("utf-8")
                if not s == "$EndMeshFormat":
                    raise ValueError("Expected $EndMeshFormat but got {0}".format(s))
                continue
            if s == "$Nodes":
                s = f.readline().rstrip().decode("utf-8")
                try:
                    number_of_vertices = int(s)
                except:
                    raise ValueError("Expected integer, got {0}".format(s))

                all_vertices = np.zeros((3, number_of_vertices), dtype="float64")
                all_vertex_ids = np.zeros(number_of_vertices, dtype="uint32")

                if binary:
                    n = number_of_vertices * (4 + 3 * data_size)
                    s = f.read(n)
                    dtype = [("id", np.int32), ("vertex", np.float64, (3,))]
                    vertex_data = np.frombuffer(s, dtype=dtype)
                    all_vertices[:, :] = vertex_data["vertex"].T
                    all_vertex_ids[:] = vertex_data["id"]
                    s = f.readline().rstrip().decode("utf-8")  # Read end of line
                    s = f.readline().rstrip().decode("utf-8")  # Read EndNodes
                else:
                    count = 0
                    s = f.readline().rstrip()
                    while s != "$EndNodes":
                        index, x, y, z = read_vertex(s)
                        all_vertices[:, count] = (x, y, z)
                        all_vertex_ids[count] = index
                        count += 1
                        s = f.readline().rstrip().decode("utf-8")
                        if count == number_of_vertices:
                            break
                    if count != number_of_vertices:
                        raise ValueError(
                            "Expected %i vertices but got %i vertices.",
                            number_of_vertices,
                            count,
                        )
                if s != "$EndNodes":
                    raise ValueError("Expected $EndNodes but got %s.", s)
            if s == "$Elements":
                s = f.readline().rstrip().decode("utf-8")
                try:
                    number_of_elements = int(s)
                except:
                    raise ValueError("Expected integer, got %s", s)
                elements_list = []
                element_ids_list = []
                domain_indices_list = []

                if binary:
                    import struct

                    count = 0
                    while count < number_of_elements:
                        elem_header = struct.unpack("iii", f.read(3 * 4))
                        elem_type = elem_header[0]
                        nelements = elem_header[1]
                        ntags = elem_header[2]
                        nnodes = _nodes_per_elem_type[elem_type]
                        for _ in range(nelements):
                            index = struct.unpack("i", f.read(4))[0]
                            tags = struct.unpack(ntags * "i", f.read(4 * ntags))
                            nodes = struct.unpack(nnodes * "i", f.read(4 * nnodes))
                            if elem_type == 2:
                                element_ids_list.append(index)
                                elements_list.append(nodes)
                                domain_indices_list.append(tags[0])
                            count += 1
                    # Get the readline at the end of the elements
                    s = f.readline().rstrip().decode("utf-8")
                    # Read EndElements
                    s = f.readline().rstrip().decode("utf-8")
                else:
                    count = 0
                    s = f.readline().rstrip().decode("utf-8")
                    while s != "$EndElements":
                        elem = read_element(s)
                        if elem is not None:
                            index, v1, v2, v3, phys_id = elem
                            elements_list.append([v1, v2, v3])
                            domain_indices_list.append(phys_id)
                            element_ids_list.append(index)
                        count += 1
                        s = f.readline().rstrip().decode("utf-8")
                        if count == number_of_elements:
                            break
                    if count != number_of_elements:
                        raise ValueError(
                            "Expected %i elements but got %i elements.",
                            number_of_elements,
                            count,
                        )
                if s != "$EndElements":
                    raise ValueError("Expected $EndElements but got %s.", s)
        return (
            all_vertices,
            all_vertex_ids,
            elements_list,
            element_ids_list,
            domain_indices_list,
        )


def export_data_sets(
    file_name,
    grid_data_sets,
    description="",
    binary=None,
    vertex_ids=None,
    element_ids=None,
):
    """
    Export data sets to Gmsh.

    For each data set a new Gmsh file is created and named
    consecutively file_name +'_0', file_name + '1', etc.

    """
    import bempp.api
    import struct

    if binary is None:
        binary = bempp.api.GLOBAL_PARAMETERS.output.gmsh_use_binary

    def write_data(f, id_transformation, data, timesteps, data_type="vertex"):
        def write_actual_data(identifier, data_array, time_value, time_step):
            ncomp = data_array.shape[0]
            nvals = data_array.shape[1]
            if data_type == "vertex":
                f.write("$NodeData\n".encode("utf-8"))
            else:
                f.write("$ElementData\n".encode("utf-8"))
            f.write("1\n".encode("utf-8"))
            f.write(('"' + identifier + '"' + "\n").encode("utf-8"))
            f.write("1\n".encode("utf-8"))
            f.write((str(time_value) + "\n").encode("utf-8"))
            f.write(
                (
                    "3\n"
                    + str(time_step)
                    + "\n"
                    + str(ncomp)
                    + "\n"
                    + str(nvals)
                    + "\n"
                ).encode("utf-8")
            )
            for index in range(nvals):
                vals = data_array[:, index]
                if binary:
                    f.write(
                        struct.pack("<i" + ncomp * "d", id_transformation[index], *vals)
                    )
                else:
                    output_string = str(id_transformation[index])
                    for val in vals:
                        output_string += " " + str(val)
                    output_string += "\n"
                    f.write(output_string.encode("utf-8"))
            if binary:
                f.write("\n".encode("utf-8"))
            if data_type == "vertex":
                f.write("$EndNodeData\n".encode("utf-8"))
            else:
                f.write("$EndElementData\n".encode("utf-8"))

        # Write the real time steps
        if data.imag is None:
            identifier = data.description.rstrip()
        else:
            identifier = data.description.rstrip() + " (real)"

        for index in range(data.number_of_arrays):
            write_actual_data(identifier, data.real[index], timesteps[index], index)

        # Write the imaginary time steps
        if data.imag is not None:
            identifier = data.description.rstrip() + " (imag)"
            for index in range(data.number_of_arrays):
                write_actual_data(
                    identifier, data.imag[index], data.timesteps[index], index
                )

    nsets = len(grid_data_sets)
    if nsets == 1:
        extensions = [""]
    elif nsets > 1:
        extensions = [str(index) for index in range(nsets)]
    else:
        raise ValueError("'grid_data_sets' must be a list with at least one element.")

    if binary:
        import sys

        if sys.byteorder != "little":
            raise ValueError(
                "Binary export only support on little endian architectures."
            )

    for data_set_index, data_set in enumerate(grid_data_sets):
        with open(file_name + extensions[data_set_index], "wb") as f:
            # Write header
            f.write("$MeshFormat\n".encode("utf-8"))
            if binary:
                f.write("2.2 1 8\n".encode("utf-8"))
                f.write(struct.pack("i", 1))
                f.write("\n".encode("utf-8"))
            else:
                f.write("2.2 0 8\n".encode("utf-8"))
            f.write("$EndMeshFormat\n".encode("utf-8"))
            # Write nodes
            f.write("$Nodes\n".encode("utf-8"))
            nvertices = data_set.grid.number_of_vertices
            vertices = data_set.grid.vertices
            if vertex_ids is None:
                vertex_ids = range(1, nvertices + 1)
            nvertices_str = str(nvertices) + "\n"
            f.write(nvertices_str.encode("utf-8"))
            for index in range(nvertices):
                if binary:
                    # The little-endian < requirement explicity
                    # removes padding introduced by the
                    # struct module in Python
                    f.write(
                        struct.pack(
                            "<iddd",
                            vertex_ids[index],
                            vertices[0, index],
                            vertices[1, index],
                            vertices[2, index],
                        )
                    )
                else:
                    f.write(
                        "{0} {1} {2} {3}\n".format(
                            vertex_ids[index],
                            vertices[0, index],
                            vertices[1, index],
                            vertices[2, index],
                        ).encode("utf-8")
                    )
            if binary:
                f.write("\n".encode("utf-8"))
            f.write("$EndNodes\n".encode("utf-8"))
            # Write elements
            f.write("$Elements\n".encode("utf-8"))
            nelements = data_set.grid.number_of_elements
            elements = data_set.grid.elements
            if element_ids is None:
                element_ids = range(1, nelements + 1)
            domain_indices = data_set.grid.domain_indices
            domain_indices = data_set.grid.domain_indices
            nelements_str = str(nelements) + "\n"
            f.write(nelements_str.encode("utf-8"))
            if binary:
                # Write header for elements
                f.write(struct.pack("iii", 2, nelements, 2))
                # Now write elements
                for index in range(nelements):
                    f.write(
                        struct.pack(
                            6 * "i",
                            element_ids[index],
                            domain_indices[index],
                            0,
                            vertex_ids[elements[0, index]],  # Need to add offset
                            vertex_ids[
                                elements[1, index]
                            ],  # since counting in Gmsh starts at 1
                            vertex_ids[elements[2, index]],
                        )
                    )
                f.write("\n".encode("utf-8"))
            else:
                for index in range(nelements):
                    v0 = vertex_ids[elements[0, index]]
                    v1 = vertex_ids[elements[1, index]]
                    v2 = vertex_ids[elements[2, index]]
                    domain_index = domain_indices[index]
                    out_string = (
                        str(element_ids[index])
                        + " "
                        + "2"
                        + " "
                        + "2 "
                        + str(domain_index)
                        + " "
                        + "0 "
                        + str(v0)
                        + " "
                        + str(v1)
                        + " "
                        + str(v2)
                        + "\n"
                    )
                    f.write(out_string.encode("utf-8"))
            f.write("$EndElements\n".encode("utf-8"))

            # Now write out the data arrays

            if data_set.vertex_data:
                for data in data_set.vertex_data:
                    write_data(f, vertex_ids, data, data_set.timesteps, "vertex")
            if data_set.element_data:
                for data in data_set.element_data:
                    write_data(f, element_ids, data, data_set.timesteps, "element")


def create_grid_structure(
    all_vertices, all_vertex_ids, elements_list, element_ids_list, domain_indices_list
):
    """Take the parsed Gmsh data, and create the grid structure."""
    from bempp.api.file_interfaces.general_interface import GenericGrid
    import numpy as np

    # Check that vertices start with 1 and are contiguously sorted.
    if all_vertex_ids[0] != 1 or not np.all(np.diff(all_vertex_ids) == 1):
        raise ValueError("Vertices must be contiguously indexed starting from 1.")

    # We need to sort through the vertices and elements and only
    # keep those vertices that are associated with the boundary
    # elements

    vertex_used = -1 * np.ones(all_vertices.shape[1], dtype="int")
    elements = np.zeros((3, len(elements_list)), dtype="uint32")
    vertex_count = 0
    for element_index, elem in enumerate(elements_list):
        for vertex_index, vertex in enumerate(elem):
            if vertex_used[vertex - 1] == -1:
                vertex_used[vertex - 1] = vertex_count
                vertex_count += 1
            elements[vertex_index, element_index] = vertex_used[vertex - 1]
    vertices = np.zeros((3, vertex_count), dtype="float64")

    # Now choose the vertices in the right order
    # np.argsort reverses a permutation

    index_set = np.argsort(vertex_used)[-vertex_count:]
    vertices[:, :] = all_vertices[:, index_set]
    vertex_ids = np.zeros(vertex_count, dtype="uint32")
    vertex_ids[:] = all_vertex_ids[index_set]
    domain_indices = np.array(domain_indices_list, dtype="uint32")
    element_ids = np.array(element_ids_list, dtype="uint32")
    return GenericGrid(vertices, elements, vertex_ids, element_ids, "", domain_indices)


def import_data_sets(file_name):
    """Import Gmsh data sets"""
    from bempp.api.file_interfaces.general_interface import GridDataSet

    (
        all_vertices,
        all_vertex_ids,
        elements_list,
        element_ids_list,
        domain_indices_list,
    ) = parse_gmsh(file_name)
    grid_structure = create_grid_structure(
        all_vertices,
        all_vertex_ids,
        elements_list,
        element_ids_list,
        domain_indices_list,
    )
    data_dict = {"grid_data_sets": [GridDataSet(grid_structure)]}
    return data_dict
