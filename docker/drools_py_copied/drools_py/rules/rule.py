import abc


class Rule(abc.ABC):

    def to_rule(self):
        return