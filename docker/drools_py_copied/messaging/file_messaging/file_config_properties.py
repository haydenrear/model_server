import os
from typing import Optional

from python_di.env.base_module_config_props import BaseModuleProps
from python_di.properties.configuration_properties_decorator import configuration_properties
from drools_py.messaging.config_pros import ProducerConfigProps, ConsumerConfigProps


@configuration_properties(
    prefix_name="file_producer",
    fallback=os.path.join(os.path.dirname(__file__), 'file-config-fallback-application.yml')
)
class FileProducerConfigProperties(BaseModuleProps, ProducerConfigProps):
    base_directory: str


@configuration_properties(
    prefix_name="file_consumer",
    fallback=os.path.join(os.path.dirname(__file__), 'file-config-fallback-application.yml')
)
class FileConsumerConfigProperties(BaseModuleProps, ConsumerConfigProps):
    base_directory: str
    metadata_directory: str
    message_directory: str
    timeout: int
    max_batch: int
    delete_files_after_receiving: bool
    move_after_receiving: Optional[str]
