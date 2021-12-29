import os
import logging


def get_logger_simple(log_dir):
    #创建logger
    name = '_'.join(log_dir.split('/')[-2:])
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 创建handler, 用于写入日志
    logfile = os.path.join(log_dir, 'log.txt')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 定义输出格式，'%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 将logger添加到handler中
    logger.addHandler(fh)
    logger.addHandler(ch)

    fh.close()
    ch.close()
    return logger
