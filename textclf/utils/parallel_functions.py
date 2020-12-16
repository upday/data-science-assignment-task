import logging
import os
import numpy as np
import multiprocessing
from functools import partial
from pathos.pools import ProcessPool, ThreadPool
import time

logger = logging.getLogger(__name__)

DATE_FORMATS = ["%Y%m%d", "%Y-%m-%d"]


def is_iterable(obj):
    from collections.abc import Iterable  # drop `.abc` with Python 2.7 or lower

    return isinstance(obj, Iterable)


def split_data(data, num_workers=None, num_split=None, already_splitted=False):

    if already_splitted:
        logger.info("data is already splitted")
        return data

    len_data = len(data)

    if len_data == 0:
        return []

    if len_data < num_workers:  # avoid returning empty objects
        num_workers = len_data

    if num_split is None:
        num_split = num_workers

    data_split = np.array_split(data, num_split)
    return data_split


def parallelize(
    func,
    data,
    num_workers=None,
    use_process=True,
    num_split=None,
    already_splitted=False,
):

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    data_split = split_data(
        data,
        num_workers=num_workers,
        num_split=num_split,
        already_splitted=already_splitted,
    )

    if len(data_split) == 0:
        return []

    if use_process:
        strategy = ProcessPool
    else:
        strategy = ThreadPool

    with strategy(nodes=num_workers) as p:
        res = p.map(func, data_split)
        p.close()
        p.join()
        p.clear()

    return res


def run_on_subset(func, data_subset, **kwargs):
    return data_subset.apply(lambda row: func(row, **kwargs))


def parallelize_on_rows(func, data, num_workers=None, use_process=False, **kwargs):

    if ("DEBUG_USE_THREAD" in os.environ) and os.environ["DEBUG_USE_THREAD"] == "true":
        use_process = False
        logger.info(
            f"parallelize_on_rows --> Im forcing standard pandas apply for debug, otherwise errors everywhere"
        )
        res = data.apply(lambda row: func(row, **kwargs), axis=1)
        return [res]

    logger.info(f"parallelize_on_rows {func} start")
    time_start = time.time()

    logger.info(f"parallelize_on_rows\t--> use_process: {use_process}")
    res = parallelize(
        partial(run_on_subset, func, **kwargs), data, num_workers, use_process
    )

    time_end = time.time()
    logger.info(f"parallelize_on_rows {func} total time {time_end-time_start}")

    return res
