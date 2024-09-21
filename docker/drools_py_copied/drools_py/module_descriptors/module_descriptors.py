import threading
import typing

import torch.nn

from python_util.collections.collection_util import retrieve_all_indices_of
from python_util.collections.last_n_historical_dict import LastNHistoricalDict
from drools_py.configs.config import Config
from python_util.logger.logger import LoggerFacade
from drools_py.serialize.to_from_dict import ToFromJsonDict


class TorchModuleDescr(ToFromJsonDict):
    """
    user adds @EmitForward.forward and extends EmitForward, and when forward is called, iterate through the models,
    and create one of these, recursively, using create_external_torch_module_config_adapter and then calling
    create_config, and then adding it to this.
    """

    def __init__(self,
                 config: Config,
                 module: typing.Optional[torch.nn.Module],
                 event_time: int,
                 module_name: str):
        self.module_name = module_name
        self.event_time = event_time
        self._module = module
        self._config = config

    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, module: torch.nn.Module):
        self._module = module

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Config):
        self._config = config

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_self_dictionary(),
            "event_time": self.event_time,
            "module_name": self.module_name
        }

    @classmethod
    def from_dict(cls, s: dict):
        return TorchModuleDescr(
            Config.from_dict(s["config"]),
            None,
            s["event_time"],
            s["module_name"]
        )


T = typing.TypeVar("T", covariant=True, bound=ToFromJsonDict)


class DescriptorTree(typing.Generic[T]):
    def __init__(self,
                 module_id: str,
                 module: TorchModuleDescr,
                 sub_modules_hist: LastNHistoricalDict[T]):
        self.sub_modules_hist = sub_modules_hist
        self.module_id = module_id
        self.module = module


LayerExec = str
ToFromDictT = typing.TypeVar("ToFromDictT", covariant=True, bound=ToFromJsonDict)


def last_n_to_dict_to_from_dict(last_n: LastNHistoricalDict[ToFromJsonDict]) -> dict:
    return last_n_to_dict_cb(last_n, lambda x: x.to_dict())


def last_n_from_dict_to_from_dict(from_dict: dict, t: typing.Type[ToFromDictT]) -> LastNHistoricalDict[ToFromDictT]:
    return last_n_from_dict_cb(from_dict, lambda x: t.from_dict(x))


def last_n_to_dict_cb(last_n: LastNHistoricalDict, cb: typing.Callable) -> dict:
    out = {}
    out["n_key"] = last_n.max_size_keys
    out["n_lst"] = last_n.max_size_lst
    data = {}
    for k, v in last_n.dict.items():
        data[k] = [
            cb(v) for v in v
        ]
    out["data"] = data
    out["prev_key"] = last_n.prev_key
    out["next_key"] = last_n.next_key
    return out


def last_n_from_dict_cb(from_dict: dict, cb: typing.Callable) -> LastNHistoricalDict[ToFromDictT]:
    last_n = LastNHistoricalDict(from_dict["n_key"], from_dict["n_lst"])
    for k, v in from_dict["data"].items():
        last_n.dict[int(k)] = [cb(v) for v in v]

    last_n.prev_key = from_dict["prev_key"]
    last_n.next_key = from_dict["next_key"]
    return last_n


class LayerExecsTree(DescriptorTree[LayerExec], ToFromJsonDict):

    def __init__(self, module_id: str,
                 module: TorchModuleDescr,
                 sub_modules_hist: LastNHistoricalDict[LayerExec]):
        super().__init__(module_id, module, sub_modules_hist)

    def to_dict(self) -> dict:
        return {
            "module_id": self.module_id,
            "submodules_hist": last_n_to_dict_cb(self.sub_modules_hist, self.deserialize),
            "module": self.module.to_dict()
        }

    @classmethod
    def deserialize(cls, x):
        return x

    @classmethod
    def from_dict(cls, val: dict):
        return LayerExecsTree(
            val["module_id"],
            TorchModuleDescr.from_dict(val["module"]),
            last_n_from_dict_cb(val["submodules_hist"], cls.deserialize)
        )


Timestep = int


class ExecutionLayerTree:
    def __init__(self):
        self._execs: dict[Timestep, LayerExecsTree] = {}

    def __getitem__(self, item: Timestep) -> LayerExecsTree:
        return self._execs[item]

    def register(self, i: Timestep, layer_execs_tree: LayerExecsTree):
        self._execs[i] = layer_execs_tree

    def execution_order(self) -> list[str]:
        return self._create_execution_layer_tree_iter(
            lambda out, t, l: out.extend([v for v in l.sub_modules_hist.current_values()])
        )

    def _create_execution_layer_tree_iter(self, c: typing.Callable[[list, Timestep, LayerExecsTree], None]):
        out = []
        for t, l in self._execs.items():
            c(out, t, l)

        return out

    def items(self):
        return self._execs.items()

    def __len__(self):
        return len(self.execution_order())

    def executions_with_steps(self) -> list[tuple[str, Timestep]]:
        return self._create_execution_layer_tree_iter(
            lambda out, t, l: out.extend([(v, t) for v in l.sub_modules_hist.current_values()]))

    def to_dict(self) -> dict:
        return {
            "execs": {
                t: l.to_dict() for t, l in self._execs.items()
            }
        }

    @classmethod
    def from_dict(cls, value: dict):
        e = ExecutionLayerTree()
        for i, v in value["execs"].items():
            LoggerFacade.debug(f"Setting {i} to {v}")
            e._execs[i] = LayerExecsTree.from_dict(v)
        return e


class ModuleDescriptorTree(DescriptorTree[TorchModuleDescr]):
    def __init__(self,
                 module_id: str,
                 module: TorchModuleDescr,
                 input_embed: TorchModuleDescr,
                 sub_modules_hist: LastNHistoricalDict[TorchModuleDescr],
                 layer_execution_tree: typing.Optional[ExecutionLayerTree] = None):
        super().__init__(module_id, module, sub_modules_hist)
        self.input_embed = input_embed
        self.sub_modules_hist = sub_modules_hist
        self.module_id = module_id
        self.module = module
        if layer_execution_tree is not None:
            self.execution_tree = layer_execution_tree
        else:
            self.execution_tree: ExecutionLayerTree = self.create_subtree_refs(sub_modules_hist)

    def get_config_id(self, config: typing.Optional[Config]) -> typing.Optional[str]:
        from drools_py.torch_utils.torch_prov_mod_configs import MetaConfig
        if config is None:
            return None
        elif isinstance(config, MetaConfig):
            return config.class_type
        else:
            return config.__class__.__name__

    def retrieve_config_types_idx(self) -> dict[str, list[int]]:
        hist_idx = {
            m.module_name: self.get_config_id(m.config)
            for m in self.sub_modules_hist.current_values()
            if m is not None
        }
        out_idx = {}
        for i, module_name in enumerate(self.execution_tree.execution_order()):
            config_type = hist_idx[module_name]
            if config_type in out_idx.keys():
                out_idx[config_type].append(i)
            else:
                out_idx[config_type] = [i]

        return out_idx

    def base_name(self, name: str):
        if len(name) > 0:
            b = '.'.join(name.split('.')[:-1])
            if b == '.' or b == '' or b is None:
                return name
            return b
        return name

    def create_subtree_refs(self, sub_modules_hist) -> ExecutionLayerTree:
        out: ExecutionLayerTree = ExecutionLayerTree()
        computation_graph_position_counter = 0
        actual_counter = 0
        prev_base_name = None
        for i, s in enumerate(sub_modules_hist.current_values()):
            base_name = self.base_name(s.module_name)
            if prev_base_name == base_name or (i <= computation_graph_position_counter
                                               and i != 0 != computation_graph_position_counter):
                continue
            prev_base_name = base_name
            layer_exec_dict: LastNHistoricalDict[LayerExec] = LastNHistoricalDict(1)
            out.register(actual_counter, LayerExecsTree(base_name, s, layer_exec_dict))
            layer_exec_dict.insert(s.module_name)
            for j, s_1 in enumerate(sub_modules_hist.current_values()):
                if j <= i:
                    continue
                s_1: TorchModuleDescr = s_1
                name = self.base_name(s_1.module_name)
                if name == base_name:
                    layer_exec_dict.insert(s_1.module_name)
                else:
                    # can be multiple layer exec with same base name, because capturing ordering most important.
                    break
                computation_graph_position_counter = max(computation_graph_position_counter, j)
            actual_counter += 1
        return out

    def to_dict(self) -> dict:
        return {
            "module_id": self.module_id,
            "module": self.module.to_dict(),
            "submodules_hist": last_n_to_dict_to_from_dict(self.sub_modules_hist),
            "execution_tree": self.execution_tree.to_dict(),
            "input_embed": self.input_embed.to_dict()
            if self.input_embed is not None else None
        }

    @classmethod
    def from_dict(cls, val: dict):
        return ModuleDescriptorTree(
            val["module_id"],
            TorchModuleDescr.from_dict(val["module"]),
            TorchModuleDescr.from_dict(val["input_embed"])
            if 'input_embed' in val.keys() and val['input_embed'] else None,
            last_n_from_dict_to_from_dict(val["submodules_hist"], TorchModuleDescr),
            ExecutionLayerTree.from_dict(val["execution_tree"]),
        )


class ModuleActivityTree:
    def __init__(self,
                 name: str,
                 topic: str,
                 lock: threading.Lock,
                 module: ModuleDescriptorTree):
        self.module = module
        self.name = name
        self.topic = topic
        self.lock = lock

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "topic": self.topic,
            "module": self.module.to_dict()
        }

    @classmethod
    def from_dict(cls, val: dict):
        return ModuleActivityTree(
            val["name"],
            val["topic"],
            threading.Lock(),
            ModuleDescriptorTree.from_dict(val["module"])
        )

    def build_layer_type_idx(self) -> dict[str, list[int]]:
        starting = {}

        from drools_py.torch_utils.torch_prov_mod_configs import MetaConfig
        exec_orders = self.module.execution_tree.execution_order()
        for start in self.module.sub_modules_hist.current_values():
            LoggerFacade.debug(f"{start} is the next submodule")
            config = start.config
            if config is None:
                LoggerFacade.error(f"Config for {start.module_name} was None.")
                continue
            if isinstance(config, MetaConfig):
                if config.class_type not in starting.keys():
                    starting[config.class_type] = retrieve_all_indices_of(exec_orders, start.module_name)
                else:
                    starting[config.class_type].extend(retrieve_all_indices_of(exec_orders, start.module_name))
            else:
                if config.__class__.__name__ not in starting.keys():
                    starting[config.__class__.__name__] = retrieve_all_indices_of(exec_orders, start.module_name)
                else:
                    starting[config.__class__.__name__].extend(retrieve_all_indices_of(exec_orders, start.module_name))

        return starting



def create_torch_module_descr(name, mod):
    from drools_py.torch_utils.torch_prov_mod_configs import MetaConfig
    return TorchModuleDescr(
        MetaConfig(mod.__class__.__name__), mod, 10, name
    )


def insert_params(mod: torch.nn.Module,
                  last_n: LastNHistoricalDict[TorchModuleDescr]):
    for n, m in mod.named_modules():
        if m != mod:
            last_n.insert(create_torch_module_descr(n, m))
