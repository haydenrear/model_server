import abc
import asyncio
import typing

from drools_py.messaging.consumer import Consumer, ConsumerMessageT, AsyncConsumer
from drools_py.messaging.kafka_messaging.kafka_config_properties import KafkaConsumerConfigProps, KafkaConfigProps, \
    get_consumer_props

import kafka


class KafkaConsumer(AsyncConsumer, abc.ABC, typing.Generic[ConsumerMessageT]):


    def __init__(self,
                 kafka_consumer_config_props: KafkaConsumerConfigProps,
                 kafka_props: KafkaConfigProps):
        super().__init__(kafka_consumer_config_props)
        self.kafka_props = kafka_props
        self.kafka_consumer_config_props = kafka_consumer_config_props
        all_kafka = get_consumer_props(self.kafka_props, self.kafka_consumer_config_props)
        self.kafka_consumer = kafka.KafkaConsumer(**all_kafka)

    def initialize(self, topics: list[str], partitions: list[int] = None):
        self.kafka_consumer.subscribe(topics)

    def read_next_messages(self, num_read: int, timeout: int) -> list[ConsumerMessageT]:
        return [i for i in self.kafka_consumer.poll(timeout, num_read).values()]

    async def consume_message(self, message: ConsumerMessageT):
        pass

    @property
    def timeout(self) -> int:
        return self.kafka_consumer_config_props.timeout

    @property
    def max_batch(self) -> int:
        return self.kafka_consumer_config_props.max_batch

