import os

from model_server.train_conn.runner import Run
from python_di.inject.context_builder.injection_context import InjectionContext


# TODO: split into main method @python_di_application.

inject_ctx = InjectionContext()
env = inject_ctx.initialize_env()

to_scan = os.path.dirname(os.path.dirname(__file__))
inject_ctx.build_context(parent_sources={to_scan}, source_directory=os.path.dirname(to_scan))
inject_ctx.ctx.get_interface(Run)
