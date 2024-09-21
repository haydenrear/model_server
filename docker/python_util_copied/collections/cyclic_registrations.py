import abc
import asyncio
import threading
import typing
from typing import TypeVar, Generic, Callable

from python_util.logger.logger import LoggerFacade
from python_util.logger.log_level import LogLevel

T = TypeVar("T")
UnderlyingT = TypeVar("UnderlyingT", covariant=True, bound=typing.Collection)


class UnderlyingCollection(abc.ABC, Generic[T, UnderlyingT]):

    def __init__(self, coll: UnderlyingT):
        self.underlying = coll

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def add(self, to_add: T):
        pass

    @classmethod
    @abc.abstractmethod
    def create_empty(cls):
        pass

    def __eq__(self, other: T) -> bool:
        return self.underlying == other

    def __contains__(self, item: T) -> bool:
        return item in self.underlying

    def __len__(self) -> int:
        return len(self.underlying)


class UnderlyingList(UnderlyingCollection[T, list[T]], Generic[T]):

    def __init__(self, underlying: list[T]):
        UnderlyingCollection.__init__(self, underlying)
        assert isinstance(self.underlying, list)

    def clear(self):
        self.underlying.clear()

    def add(self, to_add: T):
        self.underlying.append(to_add)

    @classmethod
    def create_empty(cls):
        return UnderlyingList([])


class UnderlyingSet(UnderlyingCollection[T, set[T]], Generic[T]):

    def __init__(self, underlying: set[T]):
        super().__init__(underlying)

    def clear(self):
        self.underlying.clear()

    def add(self, to_add: T):
        self.underlying.add(to_add)

    def __str__(self):
        return f'Underlying set: {self.underlying}'

    @classmethod
    def create_empty(cls):
        return UnderlyingSet(set([]))


C = TypeVar("C", bound=UnderlyingCollection, covariant=True)


class CyclicRegistrations(Generic[C, T]):
    def __init__(self, starting_values: C = None,
                 cyclical_action: Callable = None,
                 type_var=UnderlyingSet):
        self.type_var = type_var
        starting_values = self.initialize_starting(starting_values, type_var)
        self.cyclical_action = cyclical_action
        self.starting_values: C = starting_values if starting_values else type_var.create_empty()
        self.current: C = type_var.create_empty()
        self.condition = asyncio.Condition()
        self.see_current_lock: threading.Lock = threading.Lock()
        self.was_cleared: threading.Event = threading.Event()
        self.length = len(starting_values)

    def __copy__(self):
        cyclic = CyclicRegistrations(self.starting_values, self.cyclical_action, self.type_var)
        cyclic.current = self.current
        return cyclic

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    def initialize_starting(self, starting_values, type_var):
        if (starting_values is not None and not isinstance(starting_values, UnderlyingCollection)
                and isinstance(starting_values, typing.Collection)):
            LoggerFacade.warn(f'Creating underlying collection of type {type_var} because provided non-compatible '
                              f'type.')
            empty = type_var.create_empty()
            for s in starting_values:
                empty.add(s)
            starting_values = empty
        return starting_values

    def contains_value(self, value: T) -> bool:
        return value in self.starting_values

    def contains_curr_value(self, value: T) -> bool:
        with self.condition:
            return value in self.current

    async def register(self, value: T):
        async with self.condition:
            if value not in self.starting_values:
                self.starting_values.add(value)
                self.length += 1
                self.condition.notify_all()

    async def arrive(self, value: T):
        async with self.condition:
            self.see_current_lock.acquire()
            value_not_in_current = value not in self.current
            if value_not_in_current and value in self.starting_values:
                self.current.add(value)
                self.condition.notify_all()
                # if there's nobody waiting, then clear and perform action if can clear.
            self.see_current_lock.release()

    def clear(self):
        self.see_current_lock.acquire()
        if self.current == self.starting_values:
            if self.cyclical_action is not None:
                self.cyclical_action()
            self.current.clear()
            if self.condition.locked():
                self.condition.notify_all()
        self.see_current_lock.release()

    def is_ready(self):
        self.see_current_lock.acquire()
        is_ready = self.current == self.starting_values
        self.see_current_lock.release()
        return is_ready

    def is_all_registered(self, registrations: list[T]) -> bool:
        return self.starting_values == registrations

    def is_all_current(self):
        self.see_current_lock.acquire()
        is_all_current = self.starting_values == self.current
        self.see_current_lock.release()
        return is_all_current

    async def await_current(self, do_clear: bool = True):
        current = len(self.current)
        async with self.condition:
            if LogLevel.is_debug_enabled():
                if self.current != self.starting_values:
                    LoggerFacade.debug(f'Awaiting current where {self.current} is current and {self.starting_values} '
                                       f'is starting values.')
            while not self.current == self.starting_values:
                # mutation of starting values covered by condition, and cover case where clear gets called between
                # iteration of while loop, causing continued wait.
                if current <= len(self.current):
                    current = len(self.current)
                    await self.condition.wait()
            if do_clear:
                # covers case where await is deferred and then arrive full and it would deadlock.
                self.clear()


class ListCyclicRegistrations(CyclicRegistrations[list[T], T], Generic[T]):
    def __init__(self, starting_values: list[T] = None, cyclical_action: Callable = None):
        super().__init__(starting_values, cyclical_action, UnderlyingList)


class SetCyclicRegistrations(CyclicRegistrations[set[T], T], Generic[T]):
    def __init__(self, starting_values: set[T] = None, cyclical_action: Callable = None):
        super().__init__(starting_values, cyclical_action, UnderlyingSet)


