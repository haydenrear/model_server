import collections
from typing import TypeVar, Generic

import python_util.collections.collection_util

T = TypeVar('T')


class CircularBuffer(Generic[T]):
    def __init__(self, size: int, replace_messages: bool):
        self.buffer: collections.OrderedDict[int, T] = collections.OrderedDict({})
        self.size = size
        self.current_index = 0
        self.earliest_index = 0
        self.replace_messages = replace_messages

    def __len__(self):
        return len(self.buffer)

    def room(self):
        return self.size - len(self.buffer)

    def is_full(self):
        if self.current_index is None:
            return False
        else:
            return not self.is_empty() and self.current_index - self.earliest_index == self.size

    def pop(self) -> T:
        if len(self.buffer) == 0:
            return None
        return self.remove(self.earliest_index)

    def n_messages_available(self):
        return len(self.buffer)

    def is_empty(self):
        return self.current_index == self.earliest_index

    def enqueue(self, item: T):
        if self.is_full() and self.replace_messages:
            del self.buffer[self.earliest_index]  # remove earliest item if buffer is full
            self.set_earliest_idx()
        elif self.is_full():
            return False
        self.buffer[self.current_index] = item
        self.set_earliest_idx()
        self.current_index += 1
        return True

    def get_at_index(self, index):
        if (self.earliest_index is None or (index < self.earliest_index or index >= self.current_index)
                or len(self.buffer) == 0):
            return None  # indicate that buffer does not contain item with index
        return self.buffer[index]

    def remove(self, index):
        if index < self.earliest_index or index >= self.current_index:
            return None  # indicate that buffer does not contain item with index
        removed = self.buffer[index]
        del self.buffer[index]
        if index == self.earliest_index:  # If we remove the earliest item, update earliest_index
            self.set_earliest_idx()
        return removed

    def set_earliest_idx(self):
        self.earliest_index = python_util.collections.collection_util.first_key(self.buffer)
        if self.earliest_index is None:
            assert len(self.buffer) == 0
            self.earliest_index = 0

