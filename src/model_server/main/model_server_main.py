from model_server.train_conn.runner import Run

from python_di.configs.app import boot_application

@boot_application(root_dir_cls=Run)
class ModelServerApplication:
    pass
