from contextlib import contextmanager
from time import perf_counter


@contextmanager
def measure(state, field: str):
    """累加一个阶段的墙钟耗时，单位毫秒。"""
    start = perf_counter()
    try:
        yield
    finally:
        setattr(state, field, getattr(state, field) + (perf_counter() - start) * 1000)
