"""Implementation of MPI based remote operators."""
from mpi4py import MPI
import numpy as _np

MPI_SIZE = MPI.COMM_WORLD.Get_size()
MPI_RANK = MPI.COMM_WORLD.Get_rank()

COMM = MPI.COMM_WORLD

class Message:
    """
    Messages to remote manager.
    """

    def __init__(self, status, operator_tag=None, is_complex=False):

        self.status = status
        self.is_complex = is_complex
        self.operator_tag = operator_tag


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
        self._ops = {}


    def register(op, rank, tag):
        """
        Register operator for remote execution.
        """
        if rank == MPI_RANK:
            self._ops[tag] = op

    def execute(self):

        while True:

            COMM.Recv

            # MPI Receive id of operator and if real or complex data
            # MPI Receive actual data
            # Execute operator
            # MPI Send Result to rank 1

        break when shutdown message received

