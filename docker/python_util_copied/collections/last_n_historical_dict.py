from typing import TypeVar, Generic

T = TypeVar("T")

from collections import OrderedDict


class LastNHistoricalDict(Generic[T]):
    def __init__(self, max_size_keys: int, max_size_lst: int = -1):
        super().__init__()
        self.max_size_lst = max_size_lst
        self.max_size_keys = max_size_keys
        self.dict: OrderedDict[int, list[T]] = OrderedDict()
        self.next_key = 0
        self.prev_key = 0

    def update_max_size(self, to_update_to):
        self.max_size_lst = to_update_to

    def current_len(self) -> int:
        return len(self.current_values())

    def current_values(self) -> list[T]:
        if self.prev_key not in self.dict.keys():
            return []
        return self.dict[self.prev_key]

    def force_index_key(self, index: int):
        if index not in self.dict.keys():
            self.dict[index] = []
            self.prev_key = index
            self.next_key = index + 1

    def insert(self, value):
        if len(self.dict) == 0:
            self.dict = {0: [value]}
            self.prev_key = self.next_key
            self.next_key += 1
        else:
            if self.max_size_lst != -1 and len(self.dict[self.prev_key]) >= self.max_size_lst:
                self.dict[self.next_key] = [value]
                self.prev_key = self.next_key
                self.next_key += 1
            else:
                self.dict[self.prev_key].append(value)
            if len(self.dict) == self.max_size_keys + 1:
                first = next(iter(self.dict.keys()))
                del self.dict[first]


