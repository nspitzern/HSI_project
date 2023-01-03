from time import perf_counter

import numpy as np

MINUTE = 60


def timeit(num_repeats=1):
    res_list = []
    res_times = []

    def timeit_wrapper(func):

        def timed(*args, **kwargs):
            for i in range(num_repeats):
                start = perf_counter()
                res = func(*args, **kwargs)
                end = perf_counter()

                res_list.append(res)
                res_times.append((end - start))

            avg_times = np.mean(res_times)

            if avg_times > MINUTE:
                print(f'ETA: {num_repeats} repeats took {avg_times / MINUTE} minutes')
            else:
                print(f'ETA: {num_repeats} repeats took {avg_times} seconds')

            return np.array(res_list)

        return timed

    return timeit_wrapper
