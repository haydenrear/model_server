import importlib
import os
import typing
import injector
import networkx as nx
from pydantic import BaseModel

import python_util.io_utils.io
from python_di.configs.autowire import post_construct, injectable
from drools_py.configs.config import ConfigType
from python_di.env.base_module_config_props import ConfigurationProperties, BaseModuleProps
from python_di.inject.context_builder.inject_ctx import inject_context_di
from python_di.inject.injector_provider import InjectionContextInjector
from python_di.reflect_scanner.file_parser import FileParser, FileGraphSearcher
from python_util.collections.collection_util import first
from python_util.escape_util.escape_str_util import escape_empty_str_quotes
from python_util.logger.logger import LoggerFacade
from python_di.properties.configuration_properties_decorator import configuration_properties
from typing import ForwardRef, Optional

Prop = ForwardRef('ConfigOptionConfigProperty')

WRITE_COMPONENT_TEMPLATE = """
class {}({}):
    def __init__(self, config_option: {}={}):
        super().__init__({})

{}
"""

WRITE_BUILD_CONFIG_TEMPLATE = """
    @classmethod
    @bean(profile="{}", self_factory=True, scope=ProfileScope)
    def build_{}_config(cls, **kwargs):
        return cls({})
    
    @classmethod
    def build_{}_prop(cls, **kwargs):
        return {}
"""


class ConfigOptionConfigProperty(BaseModuleProps):
    config_option_type: str
    config_option_parent: typing.Union[list[str], None, str]
    config_option_default_value: Optional[str]
    config_option_value_type: Optional[str]
    create_subclasses: Optional[list[Prop]]
    build_profiles: Optional[typing.Dict[str, object]] = None
    add_to_context: Optional[bool] = None

    @classmethod
    def new_config_option_config_property(cls,
                                          config_option_type: str,
                                          config_option_parent: Optional[str],
                                          config_option_default_value: Optional[str] = "10",
                                          config_option_value_type: Optional[str] = "int",
                                          create_subclasses: Optional[list[Prop]] = None,
                                          build_profiles: Optional[typing.Dict[str, str]] = None):
        return ConfigOptionConfigProperty(**{
            "config_option_type": config_option_type,
            "config_option_parent": config_option_parent,
            "config_option_default_value": config_option_default_value,
            "config_option_value_type": config_option_value_type,
            "create_subclasses": create_subclasses,
            "build_profiles": build_profiles
        })



@configuration_properties(
    prefix_name='config_models',
    fallback=os.path.join(os.path.dirname(__file__),
                          'generated-config-models-fallback-application.yml')
)
class ConfigModelsConfigProperties(ConfigurationProperties):
    codegen_directory: str
    config_option_properties: dict[str, list[ConfigOptionConfigProperty]]
    config_type: Optional[ConfigType]
    config_model_definition_paths: list[str]

    @classmethod
    def new_config_models_config_props(cls, codegen_dir: str,
                                       config_options_properties: dict[str, list[ConfigOptionConfigProperty]],
                                       config_type: Optional[ConfigType] = None):
        return ConfigModelsConfigProperties(**{
            "codegen_directory": codegen_dir,
            "config_option_properties": config_options_properties,
            "config_type": config_type
        })

    def _get_all_new_tys(self, added: set[str] = None, to_generate=None):
        if added is None:
            assert to_generate is None
            to_generate = self.config_option_properties
            added = set([])
        for k, v in to_generate.items():
            added.add(k)
            for q in v:
                if q.create_subclasses is not None:
                    added = self._get_all_new_tys(
                        added, {q.config_option_type: q.create_subclasses})
                else:
                    added.add(q.config_option_type)
        return added

    @inject_context_di()
    def _parse_external_config_definitions(
            self, ctx: typing.Optional[InjectionContextInjector] = None
    ) -> dict[str, nx.DiGraph]:
        assert ctx is not None
        external_defs = {}
        for p in self.config_model_definition_paths:
            parser = ctx.get_interface(FileParser, scope=injector.noscope)
            imported = importlib.import_module(p)
            config_models_definition_file = imported.__file__
            LoggerFacade.info(f"Loading config models definitions from {config_models_definition_file}.")
            parsed = parser.parse(config_models_definition_file)
            assert parser is not None
            external_defs[config_models_definition_file] = parsed
        return external_defs

    def _retrieve_module_import(self, graphs: dict[str, nx.DiGraph], config_option_ty: str) -> typing.Optional[str]:
        for k, v in graphs.items():
            file_graph_found = FileGraphSearcher.find_by_class_type_name(config_option_ty, k, v)
            if file_graph_found is not None:
                file_to_import = file_graph_found[1].id_value
                file_to_import = file_to_import.replace('.py', '').replace('/', '.')
                for i in self.config_model_definition_paths:
                    if file_to_import.endswith(i):
                        return f'from {i} import {config_option_ty}'
                else:
                    LoggerFacade.error(f"Found {file_to_import} for {config_option_ty}, but none ended with {i}")

    def generate_code(self):
        generated_code_dir = self.codegen_directory

        if not os.path.exists(generated_code_dir):
            os.makedirs(generated_code_dir)
        python_util.io_utils.io.create_file(os.path.join(generated_code_dir, '__init__.py'))

        external_defs = self._parse_external_config_definitions()
        assert len(external_defs) == len(self.config_model_definition_paths)

        with open(os.path.join(generated_code_dir, 'generated_config_models.py'), 'w') as to_write:
            to_generate = self.config_option_properties
            to_write.write('import drools_py.configs.config_models\n')
            to_write.write('from python_di.configs.component import component\n')
            to_write.write('from python_di.configs.bean import bean\n')
            to_write.write('import injector\n')
            to_write.write(
                'from python_di.inject.profile_composite_injector.composite_injector import ProfileScope, profile_scope\n')
            to_write.write('import torch\n')
            for base, subclasses in to_generate.items():
                try:
                    if subclasses is not None and len(subclasses) != 0:
                        import_statement_str = self._retrieve_module_import(external_defs, base)
                        if import_statement_str is not None:
                            LoggerFacade.debug(f"Importing {import_statement_str}.")
                        self.write_config_option_to_file(
                            base, subclasses, to_write,
                            is_base_external=import_statement_str is not None,
                            import_statement_str=import_statement_str,
                            program_graphs=external_defs
                        )
                except Exception as e:
                    LoggerFacade.error(f"Error when generating {base} with subclasses {subclasses} "
                                       f"config models: \nError: {e}\n\n")

    @classmethod
    def do_import_type(cls, type_name: str, to_write):
        if len(type_name.split('.')) != len(type_name.split(' ')):
            to_write.write(f'import {".".join(type_name.split(".")[:-1])}\n')


    def write_config_option_to_file(self, bases: str, subs: list[ConfigOptionConfigProperty],
                                    to_write, parent: Optional[ConfigOptionConfigProperty] = None,
                                    is_base_external: bool = False,
                                    import_statement_str: Optional[str] = None,
                                    program_graphs: dict[str, nx.DiGraph] = None):

        for generated in subs:
            generated: ConfigOptionConfigProperty
            self._do_perform_write(bases, generated, import_statement_str, parent, program_graphs, to_write)



    def _do_perform_write(self, bases, generated, import_statement_str, parent, program_graphs, to_write):
        assert generated.config_option_default_value is not None or parent.config_option_default_value is not None
        assert generated.config_option_value_type is not None or parent.config_option_value_type is not None
        default_value = generated.config_option_default_value if generated.config_option_default_value is not None \
            else parent.config_option_default_value
        value_type = generated.config_option_value_type if generated.config_option_value_type is not None \
            else parent.config_option_value_type
        if isinstance(generated.config_option_parent, list):
            parent_import_statements = filter(lambda x: x, [
                self._retrieve_module_import(program_graphs, parent)
                for parent in generated.config_option_parent
            ])
            to_write.write('\n'.join(parent_import_statements))
            to_write.write('\n')
            base_class = ', '.join(generated.config_option_parent)
        else:
            base_class = bases if generated.config_option_parent is None else generated.config_option_parent
        if import_statement_str is not None:
            to_write.write(f'{import_statement_str}\n')
            self.do_import_type(bases, to_write)
        value_ty = self._retrieve_module_import(program_graphs, generated.config_option_value_type)
        if value_ty is not None:
            to_write.write(value_ty)
            to_write.write("\n")
        self.do_templatize_write_config(base_class, default_value, generated, to_write, value_type,
                                        generated.build_profiles, config_type=self.config_type,
                                        add_to_ctx=generated.add_to_context)
        if generated.create_subclasses is not None and len(generated.create_subclasses) != 0:
            self.write_config_option_to_file(generated.config_option_type,
                                             generated.create_subclasses, to_write,
                                             generated,
                                             program_graphs=program_graphs)

    def do_templatize_write_config(self, base_class: typing.Union[list[str], None, str],
                                   default_value, generated, to_write, value_type, build_profiles,
                                   config_type: Optional[ConfigType] = None, add_to_ctx: Optional[bool] = None):
        init_value = "config_option"
        LoggerFacade.debug(f"Generating {generated} with {default_value} as default value and {value_type} as "
                           f"value type.")
        if value_type == str or value_type == 'str':
            default_value = escape_empty_str_quotes(default_value)
        if value_type == dict or value_type == 'dict' and build_profiles is not None and len(build_profiles) != 0:
            default_value = first(build_profiles)

        build_profiles = self.extract_build_profiles(build_profiles, value_type)

        if build_profiles is not None:
            build_configs = "\n".join([WRITE_BUILD_CONFIG_TEMPLATE.format(k, k, v, k, v)
                                       for k, v in build_profiles.items()])
        else:
            build_configs = ""

        if add_to_ctx is None or (add_to_ctx is not None and add_to_ctx):
            to_write.write("@component(scope=profile_scope)")
        else:
            LoggerFacade.debug(f"Skipped injection for {generated.config_option_type}")

        if isinstance(base_class, list):
            base_class = ', '.join(base_class)

        write_component = WRITE_COMPONENT_TEMPLATE.format(generated.config_option_type, base_class, value_type,
                                                          default_value, init_value, build_configs)

        to_write.write(write_component)

    def extract_build_profiles(self, build_profiles, value_type):
        if value_type == dict or value_type == 'dict' and build_profiles is not None:
            return self.extract_build_profile_for_ty(build_profiles, value_type, dict)
        elif value_type == dict or value_type == 'list' and build_profiles is not None:
            return self.extract_build_profile_for_ty(build_profiles, value_type, list)
        else:
            return build_profiles

    def extract_build_profile_for_ty(self, build_profiles, value_type, ty):
        new_builds = {}
        for profile, b in build_profiles.items():
            if b is not None:
                assert isinstance(b, ty | str)
                if isinstance(b, str):
                    try:
                        created = eval(b)
                        new_builds[profile] = created
                    except:
                        LoggerFacade.error(f"Failed to extract dict for profile: {profile} for {value_type}.")
        for p, b in new_builds.items():
            build_profiles[p] = b

        return build_profiles
