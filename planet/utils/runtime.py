import sys
import os
import psutil


def funcname():
    return sys._getframe(1).f_code.co_name


def memory_usage():
    proc = psutil.Process(os.getpid())
    return float(proc.memory_info().rss) / (10**9)
