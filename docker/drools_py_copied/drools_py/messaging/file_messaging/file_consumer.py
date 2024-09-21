import abc
import json
import os
import shutil
import typing

from python_util.io_utils.file_dirs import recursive_dir_iter
from python_util.logger.logger import LoggerFacade
from drools_py.messaging.consumer import Consumer, ConsumerMessageT, AsyncConsumer
from drools_py.messaging.file_messaging.file_config_properties import FileConsumerConfigProperties
from drools_py.messaging.message import FileMessageBuilderFactory

SerializedMessageT = typing.TypeVar("SerializedMessageT")


class FileConsumer(AsyncConsumer[SerializedMessageT], abc.ABC, typing.Generic[SerializedMessageT]):

    def __init__(self, file_consumer_config_properties: FileConsumerConfigProperties,
                 message_builder_factory: FileMessageBuilderFactory,
                 include_metadata: bool = False,
                 metadata_dir: typing.Optional[str] = None,
                 data_subdir: typing.Optional[str] = None):
        super().__init__(file_consumer_config_properties)
        self.include_metadata = include_metadata
        self.message_builder_factory = message_builder_factory
        self.file_consumer_config_properties = file_consumer_config_properties
        if data_subdir:
            self.data_dir = os.path.join(self.file_consumer_config_properties.base_directory,
                                         self.file_consumer_config_properties.message_directory,
                                         data_subdir)
        else:
            self.data_dir = os.path.join(self.file_consumer_config_properties.base_directory,
                                         self.file_consumer_config_properties.message_directory)
        if metadata_dir:
            self.metadata_dir = os.path.join(self.file_consumer_config_properties.base_directory,
                                             self.file_consumer_config_properties.message_directory,
                                             metadata_dir)
        else:
            self.metadata_dir = os.path.join(self.file_consumer_config_properties.base_directory,
                                             self.file_consumer_config_properties.message_directory)

        assert (self.file_consumer_config_properties.delete_files_after_receiving
                or self.file_consumer_config_properties.move_after_receiving is not None)

    def initialize(self, topics: list[str], partitions: list[int]):
        pass

    def read_next_messages(self, num_read: int, timeout: int) -> list[ConsumerMessageT]:
        consumer_messages: list[ConsumerMessageT] = []
        if self.include_metadata:
            for f in os.listdir(self.metadata_dir):
                message_builder = self.message_builder_factory.create_message_builder()
                if f.endswith(".meta"):
                    with open(f, 'r') as metadata_file:
                        loaded_metadata = json.load(metadata_file)
                        message_builder.add_metadata(loaded_metadata)
                        if "metadata" in loaded_metadata and "message_dir" in loaded_metadata["metadata"]:
                            message_file = loaded_metadata["metadata"]["message_dir"]
                            if os.path.exists(message_file):
                                self.read_process_message_file(message_builder, message_file)

                    self.post_process_file(f)
                elif f.endswith(".bin"):
                    self.read_process_message_file(message_builder, f)

                consumer_messages.append(message_builder.build())
        else:
            LoggerFacade.debug(f"Searching data dir: {self.data_dir}")
            for f in recursive_dir_iter(self.data_dir):
                message_builder = self.message_builder_factory.create_message_builder()
                if os.path.basename(f).endswith(".bin") or os.path.basename(f).endswith('.tch'):
                    self.read_process_message_file(message_builder, f)
                    self.post_process_file(f)

                consumer_messages.append(message_builder.build())

        return consumer_messages

    def post_process_file(self, f):
        LoggerFacade.debug(f"Post-processing: {f}.")
        if os.path.exists(f):
            if self.file_consumer_config_properties.delete_files_after_receiving:
                os.remove(f)
            elif self.file_consumer_config_properties.move_after_receiving is not None:
                shutil.move(f, os.path.join(self.file_consumer_config_properties.move_after_receiving,
                                            os.path.basename(f)))
            else:
                LoggerFacade.warn(f"Did not set post-processing step for file consumer: {type(self)}.")

    def read_process_message_file(self, message_builder, message_file):
        with open(message_file, 'r') as message_file_created:
            message = message_file_created.read()
            message_builder.add_message(message)
        self.post_process_file(message_file)
