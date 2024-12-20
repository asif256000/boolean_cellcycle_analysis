import logging
import time
from pathlib import Path


class LogHandler:
    def __init__(self, name: str):
        self.curr_datetime = time.gmtime(time.time())
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        self.initiate_logger()
        self.set_ignore_details_flag(flag=False)

    def initiate_logger(self):
        self.set_log_folder_name("results")
        log_file_name = self.get_logfile_name()
        formatter = self.log_formatter()
        self.set_console_log_handler(log_formatter=formatter)
        self.set_file_log_handler(log_formatter=formatter, log_file_name=log_file_name)

    def set_log_folder_name(self, folder_name: str):
        Path(folder_name).mkdir(exist_ok=True)
        self.log_folder_name = folder_name

    def set_ignore_details_flag(self, flag: bool):
        self.ignore_details = flag

    def get_logfile_name(self) -> str:
        return f"{self.log_folder_name}/{time.strftime('%m%d_%H%M%S', self.curr_datetime)}.log"

    def set_current_datetime(self, datetime: time.gmtime):
        self.curr_datetime = datetime

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

    def debug(self, msg: str, detail: bool = False):
        if not (self.ignore_details and detail):
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
