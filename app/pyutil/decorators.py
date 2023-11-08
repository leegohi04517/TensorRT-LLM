import time
import asyncio
import logging

def add_time_analysis(func):
    async def process(func, *args, **params):
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **params)
        else:
            return func(*args, **params)

    async def helper(*args, **params):
        start = time.time()
        result = await process(func, *args, **params)
        end = time.time()

        logging.info(f"{func.__name__} run cost time: {end - start}s")

        return result

    return helper
