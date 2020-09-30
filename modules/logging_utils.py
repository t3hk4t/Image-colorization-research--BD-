import logging
import os
from datetime import datetime

from modules.file_utils import FileUtils

class LoggingUtils:
    def __init__(self, filename=None):
        LoggingUtils.init(filename)

    @staticmethod
    def init(filename=None):
        rootLogger = logging.getLogger()

        if filename is None:
            FileUtils.createDir('./logs')
            filename = os.path.abspath('./logs/' + datetime.now().strftime('%y-%m-%d_auto') + '.log')
            if len(rootLogger.handlers) > 0:
                if os.path.exists(filename):
                    return

        for each in rootLogger.handlers:
            rootLogger.removeHandler(each)

        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        rootLogger.level = logging.INFO #level

        fileHandler = logging.FileHandler(filename)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

    @staticmethod
    def check_and_add():
        rootLogger = logging.getLogger()
        for each in rootLogger.handlers:
            if isinstance(each, logging.FileHandler):
                if each.baseFilename.endswith('_auto.log'):
                    LoggingUtils.init()
                    return

    @staticmethod
    def info(message):
        LoggingUtils.check_and_add()
        logging.info(message)

    @staticmethod
    def error(message):
        LoggingUtils.check_and_add()
        logging.error(message)
