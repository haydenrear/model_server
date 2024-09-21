import asyncio
from typing import Optional


def run_task(task, name: Optional[str] = None):
    print(f"starting task: {name}")
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(task())

