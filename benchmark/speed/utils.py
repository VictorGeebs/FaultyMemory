import sys
import csv
import cProfile
from typing import Callable
from tqdm import tqdm
from timeit import default_timer as timer
from datetime import datetime


def timefunc(func, *args, **kwargs):
    """Time a function.

    args:
        iterations=3

    Usage example:
        timeit(myfunc, 1, b=2)
    """
    try:
        iterations = kwargs.pop("iterations")
    except KeyError:
        iterations = 3
    elapsed = sys.maxsize
    for _ in tqdm(range(iterations)):
        start = timer()
        _ = func(*args, **kwargs)
        elapsed = min(timer() - start, elapsed)
    print(("Best of {} {}(): {:.9f}".format(iterations, func.__name__, elapsed)))
    return elapsed


def profile(func, filename, device) -> Callable:
    best = timefunc(func, iterations=5)
    logging(filename, device, best)
    p = cProfile.Profile()
    p.runcall(func)
    p.print_stats(sort="tottime")
    return func


def logging(filename, device, data):
    with open(f"{filename}.csv", "a+") as f:
        writer = csv.writer(f)
        now = datetime.now()
        writer.writerow([now, device, data])
