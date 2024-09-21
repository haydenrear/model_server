import kafka
from python_di.inject.injector_provider import InjectionContext


class MediaItemConsumer:

    def __init__(self):
        brokers = InjectionContext.environment.get_property('kafka_brokers')
        media_topic = str(InjectionContext.environment.get_property('media_topic'))
        self.consumer = kafka.KafkaConsumer(media_topic, bootstrap_servers=brokers)

    # def consume(self):

