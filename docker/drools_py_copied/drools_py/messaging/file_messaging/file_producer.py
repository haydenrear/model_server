import asyncio
import concurrent.futures
import json
import os
from asyncio import Future
from typing import Optional, Dict, Callable

import injector

from python_util.io_utils.io import write_bytes_to_disk
from python_util.logger.logger import LoggerFacade
from drools_py.messaging.file_messaging.file_config_properties import FileProducerConfigProperties
from drools_py.messaging.file_messaging.message import FileMetadataMessage
from drools_py.messaging.producer import ProducerFuture, TestProducerFuture, Producer


class FileSystemProducerFuture(ProducerFuture):

    def __init__(self, future: Future):
        self.future = future

    def add_callback(self, callback, *args, **kwargs):
        self.future.add_done_callback(callback, *args, **kwargs)

    def wait(self, timeout_ms: int = -1):
        asyncio.get_event_loop().run_until_complete(self.do_wait())

    async def do_wait(self):
        await self.future

    def is_done(self):
        return self.future.done()


class FileSystemProducer(Producer):
    def __init__(self, file_producer_props: FileProducerConfigProperties,
                 write_subdirectory: str,
                 write_metadata_file: bool = False):
        super().__init__(file_producer_props)
        LoggerFacade.info(f"Creating file system producer of type {self.__class__.__name__} "
                          f"with write metadata: {write_metadata_file}.")
        self.write_metadata_file = write_metadata_file
        assert hasattr(self.producer_config, 'base_directory')
        self.file_system_path = os.path.join(self.producer_config.base_directory, write_subdirectory)
        if not os.path.exists(self.file_system_path):
            LoggerFacade.info(f"{self.file_system_path} did not exist when initializing {type(self)}. Attempting to "
                              f"create it now.")
            os.makedirs(self.file_system_path)
            assert os.path.exists(self.file_system_path)

    def send(self,
             topic: str,
             key: str,
             message: bytes,
             callback=None,
             headers: Optional[Dict[str, str]] = None,
             timestamp: Optional[int] = None) -> FileSystemProducerFuture:
        message_path = os.path.join(self.file_system_path, f'{topic}{"-" if key else ""}{key}.bin')
        LoggerFacade.debug(f'Writing to {message_path}')
        self.do_write_file(message, message_path)
        if self.write_metadata_file:
            path = os.path.join(self.file_system_path, f'{topic}{"-" if key else ""}{key}.meta')
            self.do_write_meta(
                headers, timestamp, path,
                FileMetadataMessage(timestamp, '', [message_path]).to_dict())
        future = Future()
        if callback is not None:
            future.add_done_callback(callback)
        future.set_result(True)
        return FileSystemProducerFuture(future)

    def send_callable(self,
                      topic: str,
                      key: str,
                      write_callable: Callable,
                      headers: Optional[Dict[str, str]] = None,
                      timestamp: Optional[int] = None,
                      metadata_message: dict = None) -> ProducerFuture:
        write_callable()
        path = os.path.join(self.file_system_path, f'{topic}{"-" if key else ""}{key}.meta')
        if metadata_message is not None:
            self.do_write_meta(headers, timestamp, path,
                               metadata_message)
        result = asyncio.Future()
        result.set_result(True)
        return FileSystemProducerFuture(result)

    def do_write_file(self, message: bytes, path: str):
        write_bytes_to_disk(message, path)

    def do_write_meta(self, headers: Optional[Dict[str, str]],
                            timestamp: Optional[int],
                            path: str,
                            metadata_message: dict):
        data = json.dumps({
            "headers": headers,
            "timestamp": timestamp,
            "metadata": metadata_message
        }).encode("utf-8")
        with open(path, 'wb') as meta_file:
            meta_file.write(data)

    def flush(self):
        pass


class TestFileSystemProducer(FileSystemProducer):
    def __init__(self):
        self.test_data = {}

    def send(self, topic: str, key: str, message: bytes, callback, headers: Optional[Dict[str, str]] = None,
             timestamp: Optional[int] = None) -> ProducerFuture:
        if topic in self.test_data.keys() and key in self.test_data[topic].keys():
            self.test_data[topic][key].append(message)
        elif topic not in self.test_data.keys():
            self.test_data[topic] = {key: [message]}
        elif topic in self.test_data.keys() and key not in self.test_data[topic].keys():
            self.test_data[topic][key] = [message]
        return TestProducerFuture()

    def send_callable(self, topic: str, key: str, write_callable: Callable, headers: Optional[Dict[str, str]] = None,
                      timestamp: Optional[int] = None) -> ProducerFuture:
        LoggerFacade.info(f"Sending callable: {topic}, {key}, and {timestamp}.")
        self.send(topic, key, "test_one".encode('utf-8'), headers, timestamp)
        return TestProducerFuture()

    async def do_write_file(self, message: bytes, path: str):
        pass

    async def do_write_meta(self, headers: Optional[Dict[str, str]], timestamp: Optional[int], path: str):
        pass

    def flush(self):
        pass


