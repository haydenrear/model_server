class ReadUtil:
    @staticmethod
    def read_lines(file_path: str):
        with open(file_path) as file:
            for line in file:
                yield line

    @staticmethod
    def read_all(file_path: str):
        with open(file_path, 'r') as file:
            return file.read()

    @staticmethod
    def test_install():
        print("hello")