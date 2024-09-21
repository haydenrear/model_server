import copy
import threading
from typing import TypeVar, Generic, Optional

from pygtrie import Trie

from python_util.logger.logger import LoggerFacade

T = TypeVar("T")


def has_sub_patterns(subpattern, pattern):
    len_p = len(pattern)
    len_s = len(subpattern)

    if len_s > len_p:
        return False

    for i in range(len_p - len_s + 1):
        if pattern[i:i + len_s] == subpattern:
            return True

    return False


def create_rolling_patterns(subpattern):
    all_patterns = []
    for i in range(len(subpattern)):
        patterns = []
        for j in range(len(subpattern)):
            patterns.append(subpattern[(i + j) % len(subpattern)])
        all_patterns.append(patterns)

    return all_patterns


def has_subpattern(subpattern, pattern):
    return any([has_sub_patterns(sub_pattern_item, pattern)
                for sub_pattern_item in create_rolling_patterns(subpattern)])


def is_repetitions(sub_pattern: list[int], pattern: list[int]):
    sub_pattern_str = ','.join([str(p) for p in sub_pattern])
    pattern_str = ','.join([str(p) for p in pattern])

    out = pattern_str.replace(sub_pattern_str, '|___|').split('|___|')
    if len(out) != 0:
        for out_pattern in out:
            if out_pattern != ',' and out_pattern != '' and \
                    not has_subpattern([int(i) for i in out_pattern.split(',') if len(i) != 0], sub_pattern):
                return False

    if len(out) > 0:
        pattern_str = pattern_str.replace(out[0], '', 1)
    if len(out) > 1:
        if out[-1] != ',' and out[-1] != '':
            reversed_pattern_str = ''.join(reversed([i for i in pattern_str]))
            out_ = out[-1].strip().replace(',', '')
            reversed_value_removed = reversed_pattern_str.replace(out_, '', 1)
            pattern_str = ''.join(reversed([j for j in reversed_value_removed]))

    pattern = [int(i) for i in pattern_str.split(',') if i != ',' and i != '']

    for i, num in enumerate(sub_pattern):
        for j, pattern_start in enumerate(pattern):
            if pattern_start == num:
                for q in range(len(pattern)):
                    sub_pattern_item = sub_pattern[(q + i) % (len(sub_pattern))]
                    pattern_item = pattern[(q + j) % (len(pattern))]
                    if sub_pattern_item != pattern_item:
                        return False
                return True

    return True


def is_repetition(sub_pattern: list[int], pattern: list[int]):
    for sub_pattern_item in create_rolling_patterns(sub_pattern):
        if is_repetitions(sub_pattern_item, pattern):
            return True

    return False


class PatternKeepingList(Generic[T]):
    # TODO:
    def __init__(self, max_length):
        self.max_length = max_length
        self.data = []
        self.pattern = None
        self.did_calculate = threading.Event()
        self.did_calculate.clear()
        self.pattern_done = threading.Event()
        self.next_add_idx = -1

    def add(self, element):
        """
        # TODO: when adding next element, keep the expected next element, and if it matches the skip validations.
        :param element:
        :return:
        """
        if self.pattern is not None:
            if self.pattern_done.is_set():
                return
            self.data.append(element)
            if self.next_add_idx != -1:
                if self.next_add_idx < len(self.pattern):
                    if self.pattern[self.next_add_idx] == element:
                        self.next_add_idx += 1
                        return
                else:
                    self.next_add_idx = 0
                    return self.add(element)
            if self.data[0:len(self.pattern)] != self.pattern:
                self._reset_pattern()
            elif len(self.data) > len(self.pattern):
                for i in range(len(self.pattern), len(self.data), len(self.pattern)):
                    next_data = self.data[i:min(i + len(self.pattern), len(self.data))]
                    if not all([next_data[i] == self.pattern[i] for i in range(len(next_data))]):
                        LoggerFacade.debug(f"Pattern reset: {next_data} and {self.pattern}")
                        self._reset_pattern()
                        return
            elif len(self.data) == self.max_length:
                self.data.clear()
                self.pattern_done.set()
        else:
            self.data.append(element)

    def _reset_pattern(self):
        self.pattern = None
        self.did_calculate.clear()

    def detect_pattern_of_len(self, patt):
        return len([i for i in range(len(self.data)) if self.data[i: i + len(patt)] == patt])

    def find_pattern(self):
        self.pattern = self.break_pattern(self.data)

    def break_pattern(self, to_break_pattern):
        length = len(to_break_pattern)
        result: Optional[list] = None
        value = to_break_pattern
        for pattern_length in range(1, length // 2 + 1):
            # Construct the pattern from the start of the string
            pattern = value[:pattern_length]
            found_pattern = True

            # Check if the pattern repeats throughout the string
            for i in range(pattern_length, length, pattern_length):
                if value[i:i + pattern_length] != pattern:
                    found_pattern = False
                    break

            # If the pattern repeats, update the result and break the loop
            if found_pattern:
                if not result or len(pattern) < len(result):
                    result = pattern
                break

        return result if result is not None else copy.copy(to_break_pattern)

    def get_pattern(self):
        if self.pattern_done.is_set():
            return self.pattern
        if not self.pattern:
            self.pattern = self.break_pattern(self.data)
            self.did_calculate.clear()
            return self.pattern
        else:
            if self.did_calculate.is_set():
                self.pattern = self.break_pattern(self.data)
                self.did_calculate.clear()
                return self.pattern
            else:
                return self.pattern

