"""Interfaces for exporting to VTK"""


def export_data_sets(file_name, grid_data_sets):
    """
    Export data sets to the legacy VTK format.

    For each data set a new VTK file is created and
    named consecutively file_name + '0', file_name + '1',
    etc.

    """
    import os

    def write_data_field(f, description, data_array):
        """Write a single data array to a stream f."""

        if data_array.shape[0] == 1:
            # Scalar case
            f.write("SCALARS " + description + " double\n")
            f.write("LOOKUP_TABLE default\n")
            for val in data_array.flat:
                f.write(str(val) + "\n")
        elif data_array.shape[0] == 3:
            f.write("VECTORS " + description + " double\n")
            for index in range(data_array.shape[1]):
                val = data_array[:, index]
                f.write(str(val[0]) + " " + str(val[1]) + " " + str(val[2]) + "\n")
        f.write("\n")

    def write_time_step(fname, grid_data_set, time_index):
        """Write a single time-step to a file."""

        with open(fname, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            if len(grid_data_set.description) == 0:
                description = "File Name: " + fname
            else:
                description = grid_data_set.description.rstrip()
            f.write(description + "\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            f.write("FIELD FieldData 1\n")
            f.write("TIME 1 1 double\n")
            f.write(str(grid_data_set.timesteps[time_index]) + "\n")
            f.write("POINTS {0} double\n".format(grid_data_set.grid.number_of_vertices))
            nvertices = grid_data_set.grid.number_of_vertices
            vertices = grid_data_set.grid.vertices
            for index in range(nvertices):
                f.write(
                    "{0} {1} {2}\n".format(
                        vertices[0, index], vertices[1, index], vertices[2, index]
                    )
                )
            f.write("\n")
            elements = grid_data_set.grid.elements
            nelements = grid_data_set.grid.number_of_elements
            f.write("CELLS {0} {1}\n".format(nelements, 4 * nelements))
            for index in range(nelements):
                f.write(
                    "3 {0} {1} {2}\n".format(
                        elements[0, index], elements[1, index], elements[2, index]
                    )
                )
            f.write("\n")
            f.write("CELL_TYPES {0}\n".format(nelements))
            for _ in range(nelements):
                f.write("5\n")
            f.write("\n")
            if grid_data_set.element_data:
                f.write("CELL_DATA {0}\n".format(nelements))
                for index, data in enumerate(grid_data_set.element_data):
                    if len(data.description) == 0:
                        description = "cell_data_{0}".format(index)
                    else:
                        description = description.replace(" ", "_").rstrip()
                    if data.imag is None:
                        write_data_field(f, description, data.real[time_index])
                    else:
                        write_data_field(
                            f, description + "_real", data.real[time_index]
                        )
                        write_data_field(
                            f, description + "_imag", data.imag[time_index]
                        )

            if grid_data_set.vertex_data:
                f.write("POINT_DATA {0}\n".format(nvertices))
                for index, data in enumerate(grid_data_set.vertex_data):
                    if len(data.description) == 0:
                        description = "point_data_{0}".format(index)
                    else:
                        description = description.replace(" ", "_").rstrip()
                    if data.imag is None:
                        write_data_field(f, description, data.real[time_index])
                    else:
                        write_data_field(
                            f, description + "_real", data.real[time_index]
                        )
                        write_data_field(
                            f, description + "_imag", data.imag[time_index]
                        )

    fname, extension = os.path.splitext(file_name)

    for data_set_index, data_set in enumerate(grid_data_sets):
        for time_index in range(data_set.number_of_timesteps):
            if len(grid_data_sets) == 1:
                extension = ""
            else:
                extension = "_" + str(index)
            if data_set.number_of_timesteps > 1:
                extension += "_n" + str(time_index)
            new_fname = fname + extension + ".vtk"
            write_time_step(new_fname, data_set, time_index)
