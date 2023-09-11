import ctypes
import threading
import time
from multiprocessing import active_children

KNOWN_THREAD_NAMES = ["_messaging_loop", "_processing_loop"]


def dangling_thread_decorator(f):
    def wrapper(*args, **kwargs):
        # run the test
        f(*args, **kwargs)

        # wait for cleanup
        time.sleep(0.01)

        # allow only the main thread or daemons to remain
        for thread in threading.enumerate():
            if thread.daemon or thread.name == "MainThread":
                # ignore daemon threads
                continue

            if not any(crit_name in thread.name for crit_name in KNOWN_THREAD_NAMES):
                # ignore unknown threads (e.g. pytest threads)
                continue

            raise RuntimeError(f"{thread.name} is not a daemon and alive after the test passed.")

    return wrapper


def dangling_process_decorator(f):
    def wrapper(*args, **kwargs):
        # run the test
        f(*args, **kwargs)

        # wait for cleanup
        time.sleep(0.01)

        # allow only daemon child processes to remain
        non_daemons = [proc for proc in active_children() if not proc.daemon]
        if len(non_daemons) > 0:
            for proc in non_daemons:
                proc.terminate()
            raise RuntimeError(f"Non-daemon processes are alive after the test passed: {non_daemons}")

    return wrapper


def pytest_collection_modifyitems(items):
    """Decorate all tests with detectors for dangling threads and processes."""
    for item in items:
        item.obj = dangling_thread_decorator(item.obj)
        item.obj = dangling_process_decorator(item.obj)
