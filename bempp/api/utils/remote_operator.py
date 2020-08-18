"""Implementation of MPI based remote operators."""
from bempp.api.assembly.discrete_boundary_operator import _DiscreteOperatorBase


from mpi4py import MPI
import numpy as _np

MPI_SIZE = MPI.COMM_WORLD.Get_size()
MPI_RANK = MPI.COMM_WORLD.Get_rank()

COMM = MPI.COMM_WORLD

_REMOTE_MANAGER = None


class Message:
    """
    Messages to remote manager.
    """

    def __init__(self, status, operator_tag=None, is_complex=False, nelements=None):

        self.status = status
        self.is_complex = is_complex
        self.operator_tag = operator_tag
        self.nelements = nelements


class RankCounter:
    """Simple class to give next rank for object."""

    def __init__(self):
        """Initialize rank counter."""
        self._count = 0

    def next(self):
        """Get next rank."""
        ret_val = 1 + (self._count % MPI_SIZE)
        self._count += 1
        return ret_val


class RemoteManager:
    """Manage remote worker execution."""

    def __init__(self):
        self._op_data = {}
        self._op_location = {}
        self._tags_counter = 0

    def register(self, op, rank):
        """
        Register operator for remote execution.
        """

        tag = self._tags_counter

        self._op_data[tag] = op
        self._op_location[tag] = rank

        self._tags_counter += 1

        return tag

    def execute_worker(self):
        """Only execute on workers."""

        while True:
            msg, data, tag = self.receive_data(0)

            if msg == "SHUTDOWN":
                break

            try:
                result = self._op_data[tag].weak_form() @ data
            except Exception as e:
                self.send_error("FAILED", tag)
            else:
                self.send_data("SUCCESS", 0, tag, result)

    def send_data(self, msg, dest, operator_tag=None, data=None):
        """Send data to destination rank."""
        COMM.send(
            Message(
                msg,
                operator_tag=operator_tag,
                is_complex=_np.iscomplexobj(data),
                nelements=len(data) if data is not None else None,
            ),
            dest=dest,
        )
        if msg != "SHUTDOWN":
            COMM.Send(data, dest=dest)

    def send_error(self, msg, operator_tag):
        """Send error message to master."""

        COMM.send(Message(msg, operator_tag=operator_tag), dest=0)

    def receive_data(self, source):
        """Receive data."""
        message = COMM.recv(source=source)

        if message.status == "FAILED":
            raise Exception(f"Error in worker: {source}")
        if message.status == "SHUTDOWN":
            return message.status, None, None
        if message.is_complex:
            dtype = _np.complex128
        else:
            dtype = _np.float64
        data = _np.empty(message.nelements, dtype=dtype)

        COMM.Recv(data, source=source)

        return message.status, data, message.operator_tag

    def submit_computation(self, tag, x):
        """Submit computation to operator (only execute on rank 0)."""

        rank = self._op_location[tag]

        self.send_data("COMPUTE", rank, tag, x)

    def receive_result(self, tag):
        """Receive result from a specific operator."""

        rank = self._op_location[tag]

        msg, data, _ = self.receive_data(rank)

        if msg != "SUCCESS":
            raise Exception


        return msg, data



    def shutdown(self):
        """Shutdown all workers."""

        if MPI_SIZE == 0:
            return

        for worker in range(1, MPI_SIZE):
            self.send_data("SHUTDOWN", worker)

def get_remote_manager():
    """Initialize remote manager."""
    global _REMOTE_MANAGER

    if _REMOTE_MANAGER is None:
        _REMOTE_MANAGER = RemoteManager()
    return _REMOTE_MANAGER


class RemoteBlockedDiscreteOperator(_DiscreteOperatorBase):
    """Implementation of a discrete blocked boundary operator."""

    def __init__(self, ops, is_complex=False):
        """
        Construct an operator from a two dimensional Numpy array of operators.

        ops is a list of list containing discrete boundary operators or None.
        A None entry is equivalent to a zero discrete boundary operator.

        """
        # pylint: disable=too-many-branches
        from bempp.api.utils.data_types import combined_type
        from bempp.api.assembly.discrete_boundary_operator import (
            ZeroDiscreteBoundaryOperator,
        )

        self._manager = get_remote_manager()
        self._is_complex = is_complex
        rank_counter = RankCounter()

        if not isinstance(ops, _np.ndarray):
            ops = _np.array(ops)

        rows = ops.shape[0]
        cols = ops.shape[1]

        self._ndims = (rows, cols)

        self._operators = _np.empty((rows, cols), dtype=_np.object)
        self._tags = _np.empty((rows, cols), dtype=_np.int)
        self._rows = -_np.ones(rows, dtype=int)
        self._cols = -_np.ones(cols, dtype=int)

        for i in range(rows):
            for j in range(cols):
                if ops[i, j] is None:
                    continue
                if self._rows[i] != -1:
                    if self._get_shape(ops[i, j])[0] != self._rows[i]:
                        raise ValueError(
                            "Block row {0} has incompatible ".format(i)
                            + " operator sizes."
                        )
                else:
                    self._rows[i] = self._get_shape(ops[i, j])[0]

                if self._cols[j] != -1:
                    if self._get_shape(ops[i, j])[1] != self._cols[j]:
                        raise ValueError(
                            "Block column {0} has incompatible".format(j)
                            + "operator sizes."
                        )
                else:
                    self._cols[j] = self._get_shape(ops[i, j])[1]
                self._operators[i, j] = ops[i, j]

        if not self._fill_complete():
            raise ValueError("Each row and column must contain at least one operator.")

        for i in range(rows):
            for j in range(cols):
                if self._operators[i, j] is None:
                    self._tags[i, j] = -1
                else:
                    self._tags[i, j] = self._manager.register(self._operators[i, j], rank_counter.next())

        shape = (_np.sum(self._rows), _np.sum(self._cols))

        if is_complex:
            dtype = _np.complex128
        else:
            dtype = _np.float64

        super().__init__(dtype, shape)


    def __getitem__(self, key):
        """Return the object at position (i, j)."""
        return self._operators[key]

    def _get_shape(self, op):
        """Get shape of boundary operator."""
        return (op.dual_to_range.global_dof_count, op.domain.global_dof_count)

    def _fill_complete(self):
        if (-1 in self._rows) or (-1 in self._cols):
            return False
        return True

    def _matvec(self, x):
        from bempp.api.utils.data_types import combined_type

        if not self._fill_complete():
            raise ValueError("Not all rows or columns contain operators.")

        ndims = len(x.shape)
        x = x.ravel()

        row_dim = 0
        res = _np.zeros(self.shape[0], dtype=combined_type(self.dtype, x.dtype))

        # Submit the computations
        for i in range(self._ndims[0]):
            col_dim = 0
            local_res = res[row_dim : row_dim + self._rows[i]]
            for j in range(self._ndims[1]):
                if self._tags[i, j] != -1:
                    # If self._tags[i, j] == -1 the operator is None
                    local_x = x[col_dim : col_dim + self._cols[j]]
                    self._manager.submit_computation(self._tags[i, j], local_x)
                col_dim += self._cols[j]
            row_dim += self._rows[i]

        # Get results back
        row_dim = 0
        for i in range(self._ndims[0]):
            col_dim = 0
            for j in range(self._ndims[1]):
                if self._tags[i, j] == -1: continue
                msg, remote_result = self._manager.receive_result(self._tags[i, j])
                if msg != "SUCCESS":
                    raise Exception(f"Remote computation for block {(i, j)} failed with message {msg}.")
                res[row_dim : row_dim + self._rows[i]] += remote_result
            row_dim += self._rows[i]

        if ndims == 2:
            res = res.reshape(-1, 1)

        return res

    def _get_row_dimensions(self):
        return self._rows

    def _get_column_dimensions(self):
        return self._cols

    @property
    def A(self):
        """TODO: add docstring."""
        raise NotImplementedError()

    def _transpose(self):
        """Implement the transpose."""
        raise NotImplementedError()

    def _adjoint(self):
        """Implement the adjoint."""
        raise NotImplementedError()

    row_dimensions = property(_get_row_dimensions)
    column_dimensions = property(_get_column_dimensions)



        





        
