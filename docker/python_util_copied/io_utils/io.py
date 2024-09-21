import os.path

BUFFER_SIZE = 8192  # Adjust the buffer size based on your specific requirements


def write_bytes_to_disk(data: bytes, file_path: str) -> None:
    with open(file_path, 'wb') as file:
        with memoryview(data) as data_view:
            offset = 0
            while offset < len(data_view):
                file.write(data_view[offset:offset + BUFFER_SIZE])
                offset += BUFFER_SIZE


def byte_encode_facade(value: str) -> bytes:
    return value.encode("utf-8")


def create_file(path: str, overwrite: bool=False):
    if not os.path.exists(path) or overwrite:
        open(path, 'w').close()

