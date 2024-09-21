import importlib
import os

from python_util.logger.logger import LoggerFacade
from python_di.env.base_module_config_props import BaseModuleProps
from python_di.properties.configuration_properties_decorator import configuration_properties
from drools_py.messaging.config_pros import ConsumerConfigProps, ProducerConfigProps


@configuration_properties(
    prefix_name="kafka_properties",
    fallback=os.path.join(os.path.dirname(__file__), "kafka-config-application.yml")
)
class KafkaConfigProps(BaseModuleProps):
    brokers: str


@configuration_properties(
    prefix_name="kafka_consumer_properties",
    fallback=os.path.join(os.path.dirname(__file__), "kafka-config-application.yml")
)
class KafkaConsumerConfigProps(BaseModuleProps, ConsumerConfigProps):
    timeout: int
    max_batch: int
    key_serializer: str
    value_serializer: str


@configuration_properties(
    prefix_name="kafka_producer_properties",
    fallback=os.path.join(os.path.dirname(__file__), "kafka-config-application.yml")
)
class KafkaProducerConfigProps(BaseModuleProps, ProducerConfigProps):
    key_serializer: str
    value_serializer: str


def get_producer_props(kafka_props: KafkaConfigProps, kafka_producer_props: KafkaProducerConfigProps):
    out_props = {}
    key_deserializer, value_deserializer = get_key_value_ser(kafka_producer_props)
    out_props["bootstrap_servers"] = kafka_props.brokers
    out_props["key_serializer"] = lambda val: key_deserializer(val)
    out_props["value_serializer"] = lambda val: value_deserializer(val)
    return out_props


def get_consumer_props(kafka_props: KafkaConfigProps, kafka_producer_props: KafkaConsumerConfigProps):
    out_props = {}
    key_serializer, value_serializer = get_key_value_ser(kafka_producer_props)
    out_props["bootstrap_servers"] = kafka_props.brokers
    out_props["value_deserializer"] = lambda val: value_serializer(val)
    out_props["key_deserializer"] = lambda val: key_serializer(val)
    return out_props




def get_key_value_ser(kafka_producer_props):
    key_serializer = import_build_ser(kafka_producer_props.key_serializer)
    value_serializer = import_build_ser(kafka_producer_props.value_serializer)
    return key_serializer, value_serializer


def import_build_ser(kafka_producer_props):
    splitted_ser = kafka_producer_props.split('.')
    if len(kafka_producer_props) == 0 or len(splitted_ser) == 1:
        LoggerFacade.error(f"Kafka serializer: {splitted_ser} was not correct!")
        return None
    imported = importlib.import_module('.'.join(splitted_ser[:-1]))
    if imported is None:
        LoggerFacade.warn(f"Error importing {splitted_ser}.")
    else:
        LoggerFacade.info(f"Importing {kafka_producer_props}.")
    key_serializer = imported.__dict__[splitted_ser[-1]]
    LoggerFacade.info(f"Created serializer: {key_serializer}.")
    return key_serializer
