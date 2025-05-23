import sys
import unittest

from flask import Flask

try:
    sys.path.append("/Users/hayde/IdeaProjects/drools/model_server/src")
except:
    pass

from model_server.train_conn.model_server_config import ServerRunnerConfig
from model_server.train_conn.server_runner import ServerRunner, HttpServerRunner
from python_di.configs.bean import test_inject
from python_di.configs.test import test_booter, boot_test
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn




@test_booter(scan_root_module=ServerRunnerConfig)
class ServerRunnerBoot:
    pass


app = Flask(__name__)


@boot_test(ctx=ServerRunnerBoot)
class ServerRunnerTest(unittest.TestCase):
    http: HttpServerRunner
    server_runner: ServerRunner

    @test_inject()
    @autowire_fn()
    def construct(self, http: HttpServerRunner,
                  server_runner: ServerRunner):
        ServerRunnerTest.http = http
        ServerRunnerTest.server_runner = server_runner
        # self.http.run()

    def test_server_runner_autowire(self):
        self.http.server_runner_provider.create_routes()


