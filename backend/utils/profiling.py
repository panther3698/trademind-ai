import time
import functools
import logging
import tracemalloc
import asyncio

logger = logging.getLogger("profiling")

def profile_timing(name=None):
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            label = name or func.__name__
            tracemalloc.start()
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            logger.info(f"[PROFILE] {label}: {elapsed:.4f}s | Mem: {current/1e6:.2f}MB (peak {peak/1e6:.2f}MB)")
            return result
        def sync_wrapper(*args, **kwargs):
            label = name or func.__name__
            tracemalloc.start()
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            logger.info(f"[PROFILE] {label}: {elapsed:.4f}s | Mem: {current/1e6:.2f}MB (peak {peak/1e6:.2f}MB)")
            return result
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator 