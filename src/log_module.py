import logging
import time


class LogHandler:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        log_file_name = self.get_logfile_name()
        formatter = self.log_formatter()
        self.set_console_log_handler(log_formatter=formatter)
        self.set_file_log_handler(log_formatter=formatter, log_file_name=log_file_name)

    def get_logfile_name(self) -> str:
        return f"results/{time.strftime('%m%d_%H%M%S', time.gmtime(time.time()))}.log"

    def log_formatter(self) -> logging.Formatter:
        return logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    def set_console_log_handler(self, log_formatter: logging.Formatter):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(log_formatter)
        self.logger.addHandler(ch)

    def set_file_log_handler(self, log_formatter: logging.Formatter, log_file_name: str):
        fh = logging.FileHandler(log_file_name)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_formatter)
        self.logger.addHandler(fh)

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def exception(self, msg: str):
        self.logger.exception(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        self.logger.critical(str(msg).upper())


logger = LogHandler(name="byc_dna_logs")