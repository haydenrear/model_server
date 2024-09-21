import asyncio

import injector

from drools_py.configs.config_models_config_props import ConfigModelsConfigProperties
from python_di.configs.autowire import injectable, post_construct
from python_di.configs.component import component


class CodegenGenerator:
    def __init__(self, codegen: ConfigModelsConfigProperties):
        self.codegen = codegen
        self.did_run = asyncio.Event()


    def run_codegen(self):
        self.codegen.generate_code()
        self.did_run.set()

    def did_codegen(self):
        return self.did_run.is_set()
