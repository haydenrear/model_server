import logging
import typing

from python_util.logger.log_level import LogLevel


class LoggerFacade:

    @staticmethod
    def debug(file_output, enable_debug: bool = False):
        if enable_debug:
            LogLevel.set_log_level(logging.DEBUG)
            assert LogLevel.is_debug_enabled()
        if LogLevel.is_debug_enabled() and LogLevel.is_write_enabled():
            with open('/Users/hayde/IdeaProjects/drools/feature-extractor/multi_modal/test_work/debug.log',
                      'a') as file:
                file.write(file_output)
                file.write('\n')
            print(f"Debug: {file_output}")
            logging.debug(file_output)
        elif LogLevel.is_debug_enabled():
            print(f"Debug: {file_output}")
            logging.debug(file_output)

    @staticmethod
    def debug_deferred(file_output: typing.Callable[[], str], enable_debug: bool = False):
        if enable_debug:
            LogLevel.set_log_level(logging.DEBUG)
            assert LogLevel.is_debug_enabled()
        if LogLevel.is_debug_enabled() and LogLevel.is_write_enabled():
            with open('/Users/hayde/IdeaProjects/drools/feature-extractor/multi_modal/test_work/debug.log',
                      'a') as file:
                deferred_value = file_output()
                file.write(deferred_value)
                file.write('\n')
        elif LogLevel.is_debug_enabled():
            deferred_value = file_output()
            print(f"Debug: {deferred_value}")
            logging.debug(deferred_value)

    @staticmethod
    def info(output):
        print(f"Info: {output}")
        logging.info(output)
        LoggerFacade.debug(output)
        if LogLevel.is_write_enabled():
            with open('/Users/hayde/IdeaProjects/drools/feature-extractor/multi_modal/test_work/info.log',
                      'a') as file:
                try:
                    file.write(output)
                    file.write('\n')
                except Exception as e:
                    print(f"Could not write in log: {e}")


    @staticmethod
    def error(output):
        if LogLevel.is_error_write_enabled():
            with open('/Users/hayde/IdeaProjects/drools/feature-extractor/multi_modal/test_work/error.log',
                      'a') as file:
                file.write(output)
                file.write('\n')
        print(f"Error: {output}")
        logging.error(output)

    @staticmethod
    def warn(output):
        print(f"Warning: {output}")
        logging.warning(output)
        LoggerFacade.info(output)
