from typing import Optional


class FileStreamer:
    def __init__(self, file_path: str,
                 encoding: Optional[str] = None,
                 max_bytes_in_memory: int = 1024 * 1024,
                 file_args: str = 'r',
                 split_value: str = '\n'):  # Default 1MB
        self.split_value = split_value
        self.file_args = file_args
        self.encoding = encoding
        self.file_path = file_path
        self.max_bytes_in_memory = max_bytes_in_memory
        self.current_offset = 0
        self.lines_in_memory = []
        self.load_chunk()
        self.file_size = -1
        self.file = self.open_file(file_args)

    def open_file(self, file_args):
        if 'b' not in file_args:
            return open(self.file_path, file_args, newline=self.split_value)
        else:
            return open(self.file_path, file_args)

    @property
    def max_length(self):
        count = 0
        if self.file_size == -1:
            file_opened = self.open_file(self.file_args)
            for _ in file_opened:
                count += 1
            self.file_size = count
        return self.file_size

    def reset(self):
        self.file.seek(0)
        self.load_chunk()

    def load_chunk(self, starting_offset=None):
        if starting_offset is not None:
            self.current_offset = starting_offset

        self.lines_in_memory = []
        with open(self.file_path, self.file_args) as file:
            file.seek(self.current_offset)
            chunk = file.read(self.max_bytes_in_memory)
            if 'b' in self.file_args:
                self.lines_in_memory = chunk.decode(self.encoding).split(self.split_value)
            else:
                self.lines_in_memory = chunk.split(self.split_value)


    def __next__(self):
        return next(self.file)

    def __getitem__(self, index):
        if index - self.current_offset < len(self.lines_in_memory):
            return self.lines_in_memory[index - self.current_offset]

        self.load_chunk(starting_offset=index)
        return self[index]