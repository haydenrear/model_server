from typing import Optional

from python_util.logger.logger import LoggerFacade
from drools_py.serialize.to_from_dict import ToDict


class ToRule(ToDict):

    def to_rule(self) -> dict:
        this_rule = self.to_dict()
        this_rule['rule_type'] = str(self.__class__.__name__)
        return this_rule

    def to_dict(self) -> Optional[dict]:
        try:
            this_dict = self.__dict__
        except Exception as e:
            LoggerFacade.error(f"Could not call to_dict on {self}. Need to implement ToDict for {type(self)}. "
                               f"Error: {e}")
            return None
        try:
            for key, val in this_dict.items():
                if isinstance(val, ToDict):
                    this_dict[key] = val.to_dict()
                else:
                    LoggerFacade.warn(f"Calling to_dict on object {val} not implementing to_dict.")
                    this_dict[key] = val.__dict__
            return this_dict
        except Exception as e:
            LoggerFacade.warn(f"Failed to translate {type(self)} into dict fully with error {e} Returning dict of "
                              f"size {len(this_dict)}")
            return this_dict
