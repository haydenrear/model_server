from abc import ABC
from typing import Generic, TypeVar

from python_util.logger.logger import LoggerFacade
from drools_py.subscription.circular_buffer import CircularBuffer
from drools_py.subscription.subscriber import Subscriber, AsyncSubscriber, AsyncSubscriberWithDynamicBackpressure, \
    MultiSubscriber

T = TypeVar('T')
S = TypeVar('S', bound=Subscriber, covariant=True)
MS = TypeVar('MS', bound=MultiSubscriber, covariant=True)


class Publisher(ABC, Generic[S, T]):

    def __init__(self, replace_buffer_messages: bool = False, buffer_size=100):
        self.buffer = CircularBuffer(buffer_size, replace_buffer_messages)
        self.subscribers: dict[S, int] = {}
        self.messages_to_subscribers: dict[int, list[S]] = {}  # map message indices to lists of subscribers

    def subscribe(self, subscriber: S):
        self.subscribers[subscriber] = 0  # initial index for each subscriber

    def unsubscribe(self, subscriber: S):
        if subscriber in self.subscribers:
            del self.subscribers[subscriber]

    def num_messages(self, subscribe):
        count = 0
        for key, subscribers in self.messages_to_subscribers.items():
            if subscribe in subscribers:
                count += 1
        return count

    def peek_next_message(self, subscriber: S, n: int = 0):
        """
        Peeks the next message if n = 0, or the curr - nth message.
        :param subscriber:
        :param n:
        :return:
        """
        start_index = self.subscribers[subscriber]
        return self.buffer.get_at_index(start_index - n)

    def get_messages(self, subscriber: S, num_messages: int):
        messages = []
        start_index = self.subscribers[subscriber]
        for i in range(min(num_messages, self.num_messages(subscriber))):
            message = self.buffer.get_at_index(start_index)
            if message is not None:
                messages.append(message)
                # Remove subscriber from the list of subscribers that haven't processed this message yet
                self.messages_to_subscribers[start_index].remove(subscriber)
                if not self.messages_to_subscribers[start_index]:  # if no subscribers left for this message
                    self.buffer.remove(start_index)  # remove message from buffer
                    del self.messages_to_subscribers[start_index]  # remove record of message
                start_index += 1
        self.subscribers[subscriber] = start_index  # update the last read message index
        return messages

    def _try_enqueue(self, message) -> bool:
        if not self.buffer.is_full():
            self.buffer.enqueue(message)
            return True
        return False

    def next(self, message) -> bool:
        did_enqueue = self._try_enqueue(message)
        if not did_enqueue:
            LoggerFacade.debug("Buffer is full, message not added")
        # Initially, every subscriber hasn't processed the new message
        self.messages_to_subscribers[self.buffer.current_index - 1] = list(self.subscribers.keys())
        for subscriber in self.subscribers.keys():
            num_messages = self.is_ready(message, subscriber)
            messages_to_send = self.get_messages(subscriber, num_messages)
            if messages_to_send and len(messages_to_send) == 1:
                subscriber.next_value(messages_to_send[0])
            elif len(messages_to_send) > 1:
                subscriber.next_values(messages_to_send)
        return did_enqueue


    def is_ready(self, message, subscriber: S) -> int:
        return subscriber.is_ready_num_messages()

    def is_publisher_ready_next(self, message):
        return all([self.is_ready(message, s) for s in self.subscribers])


class MultiPublisher(Publisher[MS, T], Generic[MS, T]):

    def is_ready(self, message, subscriber: MS):
        if subscriber.is_ready_for(message):
            return 1
        else:
            return 0


AsyncSubscriberT = TypeVar('AsyncSubscriberT', covariant=True, bound=AsyncSubscriberWithDynamicBackpressure)


class AsyncPublisher(Publisher[AsyncSubscriberT, T], Generic[AsyncSubscriberT, T]):

    async def next(self, message) -> bool:
        did_enqueue = self._try_enqueue(message)
        if not did_enqueue:
            LoggerFacade.debug("Buffer is full, message not added")
        # Initially, every subscriber hasn't processed the new message
        self.messages_to_subscribers[self.buffer.current_index - 1] = list(self.subscribers.keys())
        for subscriber in self.subscribers.keys():
            subscriber: AsyncSubscriberT = subscriber
            num_messages = subscriber.is_ready_num_messages()
            messages_to_send = self.get_messages(subscriber, num_messages)
            if messages_to_send and len(messages_to_send) == 1:
                await subscriber.next_value(messages_to_send[0])
            elif len(messages_to_send) > 1:
                await subscriber.next_values(messages_to_send)
        return did_enqueue


class AsyncPublisherWithBackPressure(AsyncPublisher[AsyncSubscriberT, T]):

    async def next(self, message) -> bool:
        did_enqueue = self._try_enqueue(message)
        if not did_enqueue:
            LoggerFacade.debug("Buffer is full, message not added")
        self.messages_to_subscribers[self.buffer.current_index - 1] = list(self.subscribers.keys())
        for subscriber in self.subscribers.keys():
            subscriber: AsyncSubscriberT = subscriber
            count = 0
            message_to_send = self.peek_next_message(subscriber, count)
            while message_to_send is not None:
                if not subscriber.ready_for(message_to_send):
                    continue
                if message_to_send is not None:
                    if subscriber.ready_for(message_to_send):
                        messages_to_send = self.get_messages(subscriber, 1)
                        if len(messages_to_send) != 1:
                            break
                        assert len(messages_to_send) == 1
                        if messages_to_send and len(messages_to_send) == 1:
                            assert message_to_send == messages_to_send[0]
                            await subscriber.next_value(messages_to_send[0])
                        elif messages_to_send is None or len(messages_to_send) == 0:
                            break
                count += 1
                message_to_send = self.peek_next_message(subscriber, count)
        return did_enqueue

