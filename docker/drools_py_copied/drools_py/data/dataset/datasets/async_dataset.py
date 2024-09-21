import asyncio
from typing import Generic, Optional

from torch.utils.data import Dataset as TorchDataset, DataLoader

from drools_py.data.dataset.datasets.abstract_datasets import AbstractDataset, DataT
from drools_py.subscription.circular_buffer import CircularBuffer
from drools_py.subscription.subscriber import AsyncSubscriberWithDynamicBackpressure


class AsyncDatasetSubscriber(AbstractDataset[DataT],
                             AsyncSubscriberWithDynamicBackpressure[DataT],
                             Generic[DataT]):

    @property
    def decorated(self):
        return self

    def __init__(self, n_messages_queue: int = 100, replace_messages: bool = True):
        self.dataset_subscriber_counter = 0
        self.buffer = CircularBuffer(n_messages_queue, replace_messages)
        self.has_next_value = asyncio.Condition()

    def ready_for(self, value: DataT) -> bool:
        return True

    async def next_value(self, subscription_message: DataT) -> bool:
        self.dataset_subscriber_counter += 1
        if subscription_message is not None:
            async with self.has_next_value:
                did_enqueue = self.buffer.enqueue(subscription_message)
                if did_enqueue:
                    self.has_next_value.notify_all()
                return did_enqueue

    async def next_values(self, subscription_message: [DataT]) -> list[bool]:
        return [
            await self.next_value(s) for s in subscription_message
        ]

    async def await_next_from_buffer(self):
        with self.has_next_value:
            if len(self.buffer) != 0:
                return
            else:
                await self.has_next_value.wait()

    def next_value_from_buffer(self):
        return self.buffer.pop()

    def is_ready_num_messages(self) -> int:
        return self.buffer.room()

    def __getitem__(self, idx: int):
        return self.next_value_from_buffer()

    def reset(self):
        raise NotImplementedError("Failed to reset. Async dataset is not resettable.")

    def next_item(self) -> Optional[DataT]:
        return self[0]

    def item(self, idx: Optional[int] = None) -> Optional[DataT]:
        if idx is not None:
            return self[idx]
        else:
            return self.next_item()

    def to_data_loader(self) -> DataLoader:
        return DataLoader(self)
