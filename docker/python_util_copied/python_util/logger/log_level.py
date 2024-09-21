import logging


class LogLevel:
    level = None
    do_write = False
    do_write_error = False


    @classmethod
    def is_debug_enabled(cls):
        return cls.level and cls.level == logging.DEBUG

    @classmethod
    def enable_do_write_error(cls):
        cls.do_write_error = True

    @classmethod
    def enable_do_write(cls):
        cls.do_write = True
        cls.do_write_error = True

    @classmethod
    def is_write_enabled(cls):
        return cls.do_write

    @classmethod
    def is_error_write_enabled(cls):
        return cls.do_write_error

    @classmethod
    def set_log_level(cls, name):
        cls.level = name

    @classmethod
    def get_log_level(cls):
        return cls.level
