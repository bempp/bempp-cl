"""Routines to administrate a process pool."""
import numpy as _np

# Variables used inside the workers.
_DATA = {}
_MY_ID = None

# Global variables in the host.
_USE_THREADS = None
_POOL = None

# Variables in host and workers
_NWORKERS = None
_IN_WORKER = False
_BUFFER = None


def worker(in_queue, out_queue, worker_id, buf):
    """Definition of a worker. """
    import bempp.api
    from bempp.api.utils import pool
    import traceback

    pool._MY_ID = worker_id
    pool._BUFFER = buf

    get = in_queue.get
    put = out_queue.put

    while True:

        job = get()
        if job is None:
            break
        fun, args, options = job
        try:
            if options == "NOARGS":
                result = fun()
            elif options == "MAP":
                result = fun(args)
            elif options == "STARMAP":
                result = fun(*args)
            else:
                raise ValueError("Unknown options.")
            put(result)
        except Exception:
            traceback.print_exc()

    bempp.api.flush_log()
    put("FINISHED")


def as_array(dtype, offset, shape):
    """
    Return part of the buffer as array.

    Parameters
    ----------
    dtype : Numpy dtype object
        The type of the array
    offset : int
        Start index in buffer
    nitems : int 
        Number of items of type dtype

    """
    from bempp.api.utils import pool

    nitems = _np.prod(shape)

    return _np.frombuffer(
        pool._BUFFER, dtype=dtype, count=nitems, offset=offset
    ).reshape(*shape)


def to_buffer(*args):
    """
    Save a number of numpy arrays to a buffer.

    """
    from bempp.api.utils import pool

    offset = 0
    result = []

    for arr in args:
        ar_size = arr.nbytes
        pool._BUFFER[offset : offset + ar_size] = _np.require(
            arr.flat, requirements="C"
        ).view(dtype="uint8")
        offset += ar_size
        result.append((arr.dtype, arr.shape))
    return result


def from_buffer(arrays):
    """
    Retrieve arrays from buffer.
    arrays is a list of tuples
    [(dtype1, shape1), (dtype2, shape2), ...],
    where dtype is the type of the array and shape 
    is a shape tuple.

    """
    from bempp.api.utils import pool

    result = []

    offset = 0
    for ar in arrays:
        dtype, shape = ar
        nbytes = _np.dtype(dtype).itemsize * _np.prod(shape)
        result.append(pool.as_array(dtype, offset, shape))
        offset += nbytes
    return result


class Pool(object):
    """
    A simple pool.

    This pool is different from the multiprocessing pool
    in that map operations are guaranteed to execute on
    all processes, and we can directly target individual
    processes.

    """

    def __init__(self, nworkers, buffer_size=100):
        """
        Initialise the pool.

        Parameters
        ----------
        nworkers : int
            Number of workers
        buffer_size : int
            Size of the shared memory buffer
            in MB.

        """
        from bempp.api.utils import pool
        from bempp.api.utils.pool import worker

        from multiprocessing import SimpleQueue
        from multiprocessing import get_context

        self._nworkers = nworkers

        ctx = get_context("spawn")

        pool._BUFFER = ctx.RawArray("b", buffer_size * 1024 * 1024)

        self._senders = [ctx.SimpleQueue() for _ in range(nworkers)]
        self._receivers = [ctx.SimpleQueue() for _ in range(nworkers)]

        self._workers = [
            ctx.Process(
                target=worker,
                args=(self._senders[i], self._receivers[i], i, pool._BUFFER),
            )
            for i in range(nworkers)
        ]
        for w in self._workers:
            w.daemon = True
            w.start()

    def _map_impl(self, fun, args, options):
        """Map implementation."""
        for index, arg in zip(range(self._nworkers), args):
            self._senders[index].put((fun, arg, options))
        return [self._receivers[index].get() for index in range(self._nworkers)]

    def map(self, fun, args=None):
        """Map function onto workers."""
        if args is None:
            options = "NOARGS"
            args = self._nworkers * [None]
        else:
            options = "MAP"
        return self._map_impl(fun, args, options)

    def starmap(self, fun, args):
        """Map function onto workers."""
        return self._map_impl(fun, args, "STARMAP")

    def shutdown(self):
        """Shutdown all workers."""
        for index in range(self._nworkers):
            self._senders[index].put(None)
        result = [self._receivers[index].get() for index in range(self._nworkers)]
        for worker in self._workers:
            worker.join()


def _raise_if_not_worker(name):
    """Raise exception if not in worker."""
    if not is_worker():
        raise Exception(f"Method {name} can only be called inside a worker.")


def insert_data(key, data):
    """Insert data."""
    from bempp.api.utils import pool

    _raise_if_not_worker("insert_data")
    pool._DATA[key] = data


def is_initialised():
    """Return true if pool is initialised."""
    return _POOL is not None


def get_data(key):
    """Return data."""
    from bempp.api.utils import pool

    _raise_if_not_worker("get_data")
    return pool._DATA[key]


def get_id():
    """Return my id."""
    return _MY_ID


def remove_key(key):
    """Remove data from pool."""
    from bempp.api.utils import pool

    execute(_remove_key_worker, key)


def has_key(key):
    """Return if key exists in pool data."""
    from bempp.api.utils import pool

    return key in pool._DATA


def is_worker():
    """Returns true if called from worker process."""
    return _IN_WORKER is True


def create_pool(nworkers, use_threading=False, log=True, buffer_size=100):
    """Create a pool."""

    from bempp.api.utils import pool
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp

    pool._POOL = Pool(nworkers, buffer_size=buffer_size)
    pool._NWORKERS = nworkers

    if log:
        enable_pool_log()
    starmap(_init_worker, zip(range(nworkers), nworkers * [nworkers]))


def nworkers():
    """Get number of workers."""
    return _NWORKERS


def map(fun, args):
    """Corresponds to multiprocessing map."""
    from bempp.api.utils import pool

    return _POOL.map(fun, args)


def starmap(fun, args):
    """Corresponds to multiprocessing map."""
    from bempp.api.utils import pool

    return _POOL.starmap(fun, args)


def execute(fun, *args):
    """Execute function with the same arguments on all workers."""

    if len(args) == 0:
        return _POOL.map(fun)
    else:
        return _POOL.starmap(fun, nworkers() * [args])


def enable_pool_log():
    """Enable console logging in pools."""
    execute(_enable_pool_log_worker)


def _assign_ids(nworkers):
    """Assign pool ids."""
    map(_assign_ids_worker, zip(range(nworkers), nworkers * [nworkers]))


def shutdown():
    """Shutdown the pool."""
    from bempp.api.utils import pool

    pool._POOL.shutdown()
    pool._POOL = None
    pool._NWORKERS = False
    pool._USE_THREADS = None


def _init_worker(my_id, nworkers):
    """Initialise workers."""
    from bempp.api import log
    from bempp.api.utils import pool

    pool._NWORKERS = nworkers
    pool._IN_WORKER = True
    log(f"Created worker {pool._MY_ID} out of {pool.nworkers()}.")


def _enable_pool_log_worker():
    import bempp.api

    bempp.api.enable_console_logging()


def _execute_function_without_arguments(fun):
    """Execute function without arguments."""
    return fun()


def _remove_key_worker(key):
    from bempp.api.utils import pool

    del pool._DATA[key]

