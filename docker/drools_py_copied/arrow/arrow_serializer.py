import os

import pyarrow as pa


def get_py_arrow_input_stream(arrow_dir):
    return pa.input_stream(arrow_dir)


def get_py_arrow_file_obj(arrow_dir):
    return pa.ipc.open_file(arrow_dir)


def read_torch_tensor(torch_tensor_path: str):
    arrow_data_name = get_py_arrow_input_stream(torch_tensor_path)
    arrow_tensor: pa.Tensor = pa.ipc.read_tensor(arrow_data_name)
    return arrow_tensor


def read_plain_text(plain_text_path: str):
    plain_text_path = pa.ipc.RecordBatchFileReader(pa.input_stream(plain_text_path))
    return plain_text_path

def read_message(path_in):
    return pa.ipc.read_message(get_py_arrow_input_stream(path_in))