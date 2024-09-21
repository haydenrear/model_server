import json
from typing import Optional, Dict

import injector
import kafka

from kafka.producer.future import FutureRecordMetadata

from python_di.env.base_env_properties import Environment
from drools_py.messaging.kafka_messaging.kafka_config_properties import KafkaConfigProps, KafkaProducerConfigProps, \
    get_producer_props
from drools_py.messaging.producer import ProducerFuture, Producer


class KafkaProducerFuture(ProducerFuture):

    def __init__(self, future: Optional[FutureRecordMetadata] = None):
        self.future = future

    def add_callback(self, callback, *args, **kwargs):
        if self.future:
            self.future.add_callback(callback, args, kwargs)

    def wait(self, timeout_ms: int = -1):
        if self.future:
            self.future.get(timeout_ms * 1000)


class KafkaProducer(Producer):

    def __init__(self, kafka_producer_config_props: KafkaProducerConfigProps, kafka_props: KafkaConfigProps):
        super().__init__(kafka_producer_config_props)
        self.producer = kafka.KafkaProducer(
            **get_producer_props(kafka_props, kafka_producer_config_props)
        )

    def flush(self):
        self.producer.flush()

    def send(self, topic: str, key: str, message: bytes, callback,
             headers: Optional[Dict[str, str]] = None, timestamp: Optional[int] = None) -> KafkaProducerFuture:
        future = KafkaProducerFuture(self.producer.send(topic, message, key))
        future.add_callback(callback)
        return future
