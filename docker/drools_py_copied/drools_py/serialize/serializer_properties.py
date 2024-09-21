import os

from python_di.env.base_module_config_props import ConfigurationProperties
from python_di.properties.configuration_properties_decorator import configuration_properties


@configuration_properties(
    prefix_name='serialization',
    fallback=os.path.join(os.path.dirname(__file__), 'fallback-hook-serialization-application.yml')
)
class SerializerProperties(ConfigurationProperties):
    torch_serialize_base_out_dir: str
