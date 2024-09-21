import typing

from python_util.logger.logger import LoggerFacade


def in_iterable(key, to_search):
    return any([str(key) == str(i) for i in to_search])


def deep_equals(one, two):
    if hasattr(one, '__dict__') and not hasattr(two, '__dict__'):
        LoggerFacade.debug("One has dict and two is not.")
        return False
    else:
        if isinstance(one, dict) and not isinstance(two, dict):
            LoggerFacade.debug("One is dict and two is not.")
            return False
        elif isinstance(one, dict):
            for key_, val_ in one.items():
                if not in_iterable(key_, two.keys()):
                    LoggerFacade.debug(f"One has {key_} and two does not.")
                    return False
                if not deep_equals(two[str(key_)], val_):
                    return False
            return True
        if isinstance(one, typing.Iterable):
            for iterator_type in [list, set]:
                if isinstance(one, iterator_type) and not isinstance(two, iterator_type):
                    LoggerFacade.debug(f"One is {iterator_type} and two is not.")
                    return False
                elif isinstance(one, iterator_type):
                    if len(one) != len(two):
                        LoggerFacade.debug(f"One is length {len(one)} and two is length {len(two)}.")
                        return False
            if isinstance(one, list):
                return all([deep_equals(one[i], two[i]) for i in range(len(one))])
            if isinstance(one, set):
                # already checked same size.
                for item_to_check in one:
                    if not any([deep_equals(to_check_from_two, item_to_check) for to_check_from_two in two]):
                        return False
                return True
        else:
            equals_two = one == two
            if equals_two:
                LoggerFacade.debug(f'{one} does not equal {two}.')
                return True
            else:
                two_vars = vars(two)
                for key, val in vars(one).items():
                    if key not in two_vars.keys():
                        LoggerFacade.debug(f"One has {key} and two does not.")
                        return False
                    if not deep_equals(two_vars[key], val):
                        return False
                return True

    return True
