import asyncio
import threading
from typing import Callable, Optional


class WorkQueue:
    def __init__(self,
                 has_room: Optional[Callable[[list], bool]] = None):
        self.data = []
        self.has_room = has_room
        self.condition = asyncio.Condition()

    async def await_dequeue(self):
        if len(self.data) != 0:
            return self.next()
        else:
            async with self.condition:
                await self.condition.wait()
                return self.next()

    def dequeue(self):
        if len(self.data) != 0:
            return self.next()

    def peek(self):
        if len(self.data) != 0:
            return self.peek()

    def has_next(self):
        return len(self.data) != 0

    async def async_offer(self, value):
        if not self.has_room or self.has_room(self.data):
            self.data.insert(0, value)
            async with self.condition:
                self.condition.notify_all()
            return True
        else:
            return False

    def offer(self, value):
        asyncio.get_event_loop().run_until_complete(self.async_offer(value))

    def next(self):
        return self.data.pop()