import threading
from abc import ABC, abstractmethod

from python_util.monads.util import flatmap


class WorkGroup(ABC):
    @abstractmethod
    def key(self) -> str:
        pass


class Work(WorkGroup, ABC):
    def __init__(self, work_group_key: str):
        self.work_group_key = work_group_key

    def key(self) -> str:
        return self.work_group_key


class WorkqueueThread(threading.Thread, ABC):

    def __init__(self):
        super().__init__()
        self.work: dict[str, list[Work]] = {}

    def num_work(self) -> int:
        return len(self.work)


class ThreadWorkManager(ABC):

    def __init__(self, min_work_per_thread: int, max_threads: int):
        self.threads: list[WorkqueueThread] = []
        self.min_work_per_thread = min_work_per_thread
        self.max_threads = max_threads

    def assign_work(self, work: Work):
        num_threads = len(self.threads)
        num_consumers = len(list(flatmap(lambda x: x, [processor.work.values()
                                                                for processor in self.threads])))

        work_per_thread = 0 if num_threads == 0 else num_consumers // num_threads

        if work_per_thread < self.min_work_per_thread and num_threads < self.max_threads:
            created_thread: WorkqueueThread = self.get_thread(work)
            self.threads.append(created_thread)
            created_thread.start()

        else:
            min = None
            for i in self.threads:
                if not min or i.num_work() < min.num_work():
                    min = i

            if min:
                if work.key() not in min.work.keys():
                    min.work[work.key()] = [work]
                else:
                    min.work[work.key()].append(work)

    @abstractmethod
    def get_thread(self, metric_processor: Work) -> WorkqueueThread:
        pass
