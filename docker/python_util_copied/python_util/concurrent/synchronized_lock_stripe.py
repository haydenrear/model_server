import functools
import threading
import typing

from python_util.logger.logger import LoggerFacade


CallableT = typing.TypeVar('CallableT', bound=typing.Callable)
LockId = str
DEFAULT_LOCK_KWARG_NAME = 'default_lock'

class LockStripingLocks:
    def __init__(self):
        self.locks: dict[LockId, threading.RLock] = {}


def synchronized_lock_striping(locks: LockStripingLocks, lock_arg_arg_name: str):
    """
    Creates/uses lock for each lock_arg_arg_name.
    :param locks:
    :param lock_arg_arg_name:
    :return:
    """
    def outside_wrapper(function):

        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            if lock_arg_arg_name not in kwargs.keys() or kwargs[lock_arg_arg_name] is None:
                if DEFAULT_LOCK_KWARG_NAME in locks.locks.keys():
                    with locks.locks[DEFAULT_LOCK_KWARG_NAME]:
                        return function(self, *args, **kwargs)
                else:
                    locks.locks[DEFAULT_LOCK_KWARG_NAME] = threading.RLock()
                    with locks.locks[DEFAULT_LOCK_KWARG_NAME]:
                        return function(self, *args, **kwargs)

            lock_id = kwargs[lock_arg_arg_name]
            if lock_id not in locks.locks.keys():
                LoggerFacade.debug(f"Creating new lock with id {locks}.")
                locks.locks[lock_id] = threading.RLock()
            with locks.locks[lock_id]:
                return function(self, *args, **kwargs)


        return wrapper

    return outside_wrapper
