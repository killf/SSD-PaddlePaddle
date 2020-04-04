from multiprocessing import Process, Pipe, Value
import time
import os


class StopGenerator:
    def __init__(self, pid=None):
        self.pid = pid


class DataLoader:
    def __init__(self, generator, batch_size=4, maxsize=32, collate_fn=None, shuffle=False):
        self.generator = generator
        self.batch_size = batch_size
        self.maxsize = maxsize
        self.collate_fn = collate_fn
        self.num_worker = 1
        self.shuffle = shuffle

    def __iter__(self):
        def sample_generator(generator, r, w, count):
            for item in generator:
                while count.value >= self.maxsize:
                    time.sleep(0.02)
                    continue

                w.send(item)
                with count.get_lock():
                    count.value += 1

            w.send(StopGenerator(pid=os.getpid()))
            with count.get_lock():
                count.value += 1

        r, w = Pipe(True)
        count = Value('i', 0)

        process_map = dict()
        for i in range(self.num_worker):
            process = Process(target=sample_generator, args=(self.generator, r, w, count))
            process.start()
            process_map[process.pid] = process

        def parallel_generator():
            result = []
            while len(process_map) > 0:
                item = r.recv()
                with count.get_lock():
                    count.value -= 1

                if isinstance(item, StopGenerator):
                    del process_map[item.pid]
                    continue

                result.append(item)
                if len(result) >= self.batch_size:
                    if self.collate_fn is not None:
                        result = self.collate_fn(result)

                    yield result
                    result = []

        return parallel_generator()
