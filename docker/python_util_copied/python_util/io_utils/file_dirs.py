import os
import sys
import logging
import uuid

from python_util.logger.logger import LoggerFacade


def get_dir(file, directory_name):
    if file is None or file == '/':
        return None

    if os.path.isfile(file):
        file = os.path.dirname(file)

    for directory in os.listdir(file):
        if os.path.basename(directory) == directory_name:
            return os.path.join(file, directory)

    return get_dir(os.path.dirname(file), directory_name)


def recursive_dir_iter(directory_name):
    for subdir, dirs, files in os.walk(directory_name):
        for file in files:
            yield os.path.join(subdir, file)


def delete_recursive(directory_name):
    for d in recursive_dir_iter(directory_name):
        os.remove(d)


def get_data_dir(file):
    return os.path.join(get_dir(file, 'work'), 'data')


def make_dirs(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def remove_from_dir(index_dir):
    for file in os.listdir(index_dir):
        try:
            os.remove(os.path.join(index_dir, file))
        except Exception as e:
            print(f"Could not remove: {index_dir} with exception: {e}")

    try:
        os.rmdir(index_dir)
    except Exception as e:
        print(f"Could not remove: {index_dir} with exception: {e}")


def iterate_files_in_directories(directory) -> iter:
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


def create_py_import(file, src) -> str:
    import_val = os.path.relpath(file, src) \
        .replace('../', '') \
        .replace('/', '.')

    if import_val.endswith('.py'):
        import_val = import_val[:-3]
    if import_val.startswith(os.path.basename(sys.prefix)) and 'site-packages' in import_val:
        return import_val.split('site-packages.')[1]
    elif import_val.startswith(sys.prefix):
        logging.error(f'{file} started with the environment but did not contain site-packages, which using to split '
                      f'to get the import.')

    return import_val


def get_base_path_of_current_file(call_file, num_up=0) -> str:
    next_file = call_file
    for i in range(num_up):
        next_file = os.path.dirname(next_file)
    return os.path.dirname(os.path.abspath(next_file))


def get_test_work_dir(file):
    s = get_dir(file, 'test_work')
    return s


def get_work_dir(file):
    s = get_dir(file, 'work')
    return s


def get_resources_dir(file):
    """
    Recursively gets the directory of the file and checks to see if resources exists until reaches the base directory.
    :param file:
    :return:
    """
    return get_dir(file, 'resources')


def try_remove(file):
    try:
        if os.path.isfile(file):
            os.remove(file)
        elif os.path.isdir(file):
            os.removedirs(file)
    except Exception as e:
        LoggerFacade.warn(f"Failed to delete {file} with error: {e}")


def delete_files_and_dirs_recursively(file_or_dir: str):
    if os.path.isdir(file_or_dir):
        for file_dir in os.listdir(file_or_dir):
            delete_files_and_dirs_recursively(os.path.join(file_or_dir, file_dir))
        try:
            try_remove(file_or_dir)
        except Exception as e:
            LoggerFacade.warn(f"Failed to delete {file_or_dir} with error: {e}")
    elif os.path.isfile(file_or_dir):
        try_remove(file_or_dir)


def create_random_file_name(suffix: str):
    out = (str(uuid.uuid4()).replace(":", "")
           .replace("*", "")
           .replace("?", "")
           .replace("/", ""))
    return f'{out[:min(8, len(out))]}.{suffix}'
