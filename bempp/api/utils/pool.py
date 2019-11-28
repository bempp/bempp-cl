"""Routines to administrate a process pool."""

# Variables used inside the workers.
_DATA = {}
_MY_ID = None

# Global variables in the host.
_USE_THREADS = None
_POOL = None

# Variables in host and workers
_NWORKERS = None
_IN_WORKER = None


def insert_data(key, data):
    """Insert data."""
    from bempp.api.utils import pool
    pool._DATA[key] = data

def get_data(key):
    """Return data."""
    from bempp.api.utils import pool
    return pool._DATA[key]

def create_pool(nworkers, use_threading=False, log=True):
    """Create a pool."""
    from bempp.api.utils import pool
    #from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp

    if not use_threading:
        pool._POOL = mp.get_context("spawn").Pool(nworkers)
        #pool._POOL = ProcessPoolExecutor(nworkers, mp_context=mp.get_context("spawn"))
        pool._USE_THREADS = False
    else:
        pool._POOL = ThreadPoolExecutor(nworkers)
        pool._USE_THREADS = True
    pool._NWORKERS = nworkers

    if log: enable_pool_log()
    _assign_ids(nworkers)

def nworkers():
    """Get number of workers."""
    return _NWORKERS

def map(fun, args):
    """Corresponds to multiprocessing map."""
    from bempp.api.utils import pool
    return _POOL.map(fun, args)

def starmap(fun, args):
    """Corresponds to multiprocessing starmap."""
    from bempp.api.utils import pool
    return _POOL.starmap(fun, args)

def execute(fun, *args):
    """Execute function with the same arguments on all workers.""" 
    from bempp.api.utils import pool

    if not args:
        map(_execute_function_without_arguments, nworkers() * [fun])
    else:
        starmap(fun, nworkers() * args)

def enable_pool_log():
    """Enable console logging in pools."""
    execute(_enable_pool_log_worker)

def _assign_ids(nworkers):
    """Assign pool ids."""
    starmap(_assign_ids_worker, zip(range(nworkers), nworkers * [nworkers]))

def shutdown():
    """Shutdown the pool."""
    from bempp.api.utils import pool

    pool._POOL.close()
    pool._POOL.join()
    pool._POOL = None
    pool._NWORKERS = False
    pool._USE_THREADS = None


def _assign_ids_worker(my_id, nworkers):
    """Function on worker."""
    from bempp.api import log
    from bempp.api.utils import pool

    pool._MY_ID = my_id
    pool._NWORKERS = nworkers
    pool._IN_WORKER = True
    log(f"Created worker {pool._MY_ID} out of {pool.nworkers()}.")

def _enable_pool_log_worker():
    import bempp.api
    bempp.api.enable_console_logging()

def _execute_function_without_arguments(fun):
    """Execute function without arguments."""
    fun()
