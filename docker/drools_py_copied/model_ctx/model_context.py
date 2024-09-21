import abc
import asyncio
import threading

from python_di.configs.component import component
from python_util.logger.logger import LoggerFacade
from drools_py.subscription.publisher import Publisher
from drools_py.subscription.subscriber import Subscriber, T


class ModelProperty:
    def __init__(self, key, val):
        self.key = key
        self.val = val


class ModelContextPublisher(Publisher[Subscriber, ModelProperty]):
    def __init__(self):
        super().__init__(True)


class ModelContextProcessor(Publisher[Subscriber, ModelProperty], Subscriber[ModelProperty]):

    def __init__(self, replace_buffer_messages: bool = True):
        super().__init__(replace_buffer_messages)
        self.publishers: dict[str, list[ModelContextPublisher]] = {}
        self.keys: list[ModelProperty] = []
        self.add_lock = threading.Lock()

    def next_value(self, subscription_message: ModelProperty):
        with self.add_lock:
            if subscription_message.key not in self.publishers.keys():
                LoggerFacade.error(f"Received message for property not registered: {subscription_message.key} "
                                   f"and {subscription_message.val}.")
            else:
                for publisher in self.publishers[subscription_message.key]:
                    publisher.next(subscription_message)

            self.keys.append(subscription_message)

    def next_values(self, subscription_message: [T]):
        raise NotImplementedError()

    def has_next(self) -> bool:
        return self.buffer.is_empty()

    def ready(self) -> bool:
        return True

    def is_ready_num_messages(self) -> int:
        return len(self.buffer)

    def subscribe_to_key(self, subscriber, key):
        with self.add_lock:
            publisher = ModelContextPublisher()
            if key not in self.publishers.keys():
                self.publishers[key] = [publisher]
            else:
                self.publishers[key].append(publisher)

            publisher.subscribe(subscriber)

            for key_value in self.keys:
                if key_value.key == key:
                    subscriber.next_value(key_value)


class ModelContextSubscriber(Subscriber[ModelProperty]):

    def __init__(self):
        self.completed: bool = False

    @abc.abstractmethod
    def update_arg(self, key: str, value):
        pass

    def next_value(self, subscription_message: ModelProperty):
        self.update_arg(subscription_message.key, subscription_message.val)

    def next_values(self, subscription_message: [T]):
        assert len(subscription_message) <= 1, ("Received multiple subscription messages when there should have only "
                                                "been one.")
        if len(subscription_message) == 1:
            self.next_value(subscription_message[0])

    def has_next(self) -> bool:
        return not self.completed

    def ready(self) -> bool:
        return True

    def is_ready_num_messages(self) -> int:
        return 1
