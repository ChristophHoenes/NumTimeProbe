import os
import logging
import logging.config
import threading
import time
from datetime import datetime
from pathlib import Path, PurePath
from typing import Optional

import dill


log_file_init_path = str(PurePath(__file__).parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


MAX_THREADS = 100
CLEANUP_PATIENCE = 1.0


def caching(cache_path, cache_file_name):
    cache_path_obj = Path(cache_path)
    if cache_path_obj.exists():
        cache_versions = sorted(cache_path_obj.glob('*' + cache_file_name))
        if len(cache_versions) > 0:
            latest_cache_version = cache_versions[-1]
        else:
            latest_cache_version = cache_path_obj
    else:
        logger.info("Creating cache directory (%s)...", cache_path_obj)
        cache_path_obj.mkdir(parents=True)
        latest_cache_version = cache_path_obj
    if latest_cache_version.is_file():
        logger.info("Loading from cache (%s)", latest_cache_version.name)
        with latest_cache_version.open('rb') as f:
            return dill.load(f)
    else:
        logger.info("Provided cache directory (%s) is empty.", latest_cache_version)


def clear_cache(cache_path: str = '../data/NumTabQA/.cache',
                prefix: Optional[str] = None,
                postfix: Optional[str] = None,
                keep_latest: bool = True
                ) -> None:
    cache_path_obj = Path(cache_path)
    cache_versions = sorted(cache_path_obj.glob((prefix or '') + '*' + (postfix or '')))
    for v, version in enumerate(cache_versions):
        if keep_latest and v == (len(cache_versions) - 1):
            break
        version.unlink()


def save_version(obj, cache_path, cache_file_name):
    save_path = Path(cache_path) / (datetime.now().strftime('%y%m%d_%H%M_%S_%f_')
                                    + cache_file_name)
    if (hasattr(obj, 'prepare_for_pickle') and callable(obj.prepare_for_pickle)):
        obj.prepare_for_pickle()
    logger.info(f"Writing {cache_file_name} to disk")
    with save_path.open('wb') as f:
        dill.dump(obj, f)


def delete_dataset(dataset):
    """
    function to remove intermediate dataset versions created by huggingface datasets library in
    every processing step with .map(), that can quickly deplete disk space for large datasets.
    see: https://discuss.huggingface.co/t/keeping-only-current-dataset-state-in-cache/6740
    """
    cached_files = [cache_file["filename"] for cache_file in dataset.cache_files]
    del dataset
    for cached_file in cached_files:
        os.remove(cached_file)


# timed cleanup with threads proved impractical and was solved differently with cached propery and weak reference
def timed_cleanup_thread(cleanup_func, cleanup_thread=None):
    if cleanup_thread is None:
        cleanup_thread = threading.Thread()
        cleanup_thread.start()
    if cleanup_thread.is_alive():
        # when MAX_THREADS is reached give the program some time to wait if some cleanup threads terminate before starting new ones
        # else continue registering new cleanup threads for every table
        for i in range(MAX_THREADS):
            if threading.active_count() < MAX_THREADS:
                break
            time.sleep(CLEANUP_PATIENCE * 1.05 / MAX_THREADS)
        if threading.active_count() >= MAX_THREADS:
            raise RuntimeError("Number of threads is at MAX_THREADS even after waiting for CLEANUP_PATIENCE! Fix concurency.")
        # to avoid Thread accumulation cleanup_func should also delete (the reference) the thread after execution
        cleanup_thread = threading.Timer(CLEANUP_PATIENCE, cleanup_func)
        cleanup_thread.start()
    return cleanup_thread


def timed_cleanup(obj, attribute, last_access):
    """ Generic example for a cleanup function that is called by a timer thread. """
    if time.time() - last_access >= CLEANUP_PATIENCE:
        setattr(obj, attribute, None)
    # without return threads often do not close correctly
    return
