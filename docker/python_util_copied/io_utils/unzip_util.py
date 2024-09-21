import zipfile


class ZipUtil:

    @staticmethod
    def unzip(in_file: str, out_file: str):
        with zipfile.ZipFile(in_file, 'r') as zip_file:
            zip_file.extractall(out_file)

    @staticmethod
    def zip(out_file: str, in_file: str):
        with zipfile.ZipFile(out_file, 'w') as zip_file:
            zip_file.write(in_file)
