
from multiprocessing import Queue
import queue
from threading import Thread

def process_map(func, iterator, n_proc=4, maxsize=2):
    """
    Take an `iterator` of key, value pairs and apply `func` to all values using `n_proc` processes.
    """
    if n_proc == 0: return ((k, func(v)) for k, v in iterator)
    return iter(ProcessMap(func, iterator, n_proc, output_queue=Queue(maxsize)))


class ProcessMap(Thread):

    def __init__(self, func, iterator, n_proc, output_queue=None, starmap=False, send_key=False):
        super().__init__()
        self.iterator = iterator
        self.starmap = starmap
        self.send_key = send_key
        self.work_queue = Queue(n_proc * 2)
        self.output_queue = output_queue or Queue()
        self.processes = [
            MapWorker(func, self.work_queue, self.output_queue, self.starmap, self.send_key)
            for _ in range(n_proc)
        ]

    def start(self):
        for process in self.processes:
            process.start()
        super().start()

    def run(self):
        for k, v in self.iterator:
            self.work_queue.put((k, v))
        for _ in self.processes:
            self.work_queue.put(StopIteration)
        for process in self.processes:
            process.join()
        self.output_queue.put(StopIteration)

    def __iter__(self):
        self.start()
        while True:
            item = self.output_queue.get()
            if item is StopIteration:
                break
            yield item