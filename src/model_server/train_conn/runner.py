import injector

from model_server.train_conn.server_runner import HttpServerRunner, ServerRunner
from python_di.configs.autowire import injectable
from python_di.configs.component import component


@component()
@injectable()
class Run:
    http: HttpServerRunner
    server_runner: ServerRunner

    @injector.inject
    def __init__(self,
                 http: HttpServerRunner,
                 server_runner: ServerRunner):
        self.http = http
        self.server_runner = server_runner
        self.http.run()
