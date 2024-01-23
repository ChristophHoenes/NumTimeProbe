import os
import logging
import logging.config
import shutil
import threading
import time
import warnings
from datasets import Dataset
from datetime import datetime
from pathlib import Path, PurePath
from typing import Optional, Any

import dill


log_file_init_path = str(PurePath(__file__).parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


MAX_THREADS = 100
CLEANUP_PATIENCE = 1.0


def caching(cache_path, cache_file_name) -> Optional[Any]:
    cache_path_obj = Path(cache_path)
    # check for latest version (via timestamp)
    if cache_path_obj.exists():
        cache_versions = sorted(cache_path_obj.glob(f"{cache_file_name}/*"))
        if len(cache_versions) > 0:
            latest_cache_version = cache_versions[-1]
        else:
            latest_cache_version = None
    else:
        logger.info("Creating cache directory (%s)...", cache_path_obj)
        cache_path_obj.mkdir(parents=True)
        latest_cache_version = None
    # check if the latest version containes any arrow files
    if latest_cache_version is not None:
        arrow_files = [path for path in latest_cache_version.iterdir() 
                       if path.is_file() and str(path).endswith('.arrow')
                       ]
    else:
        arrow_files = []
    # deteckt type of cached data files and load them
    pickle_target = latest_cache_version / (cache_file_name + '.pickle')
    if latest_cache_version is not None and len(arrow_files) > 0:
        logger.info("Loading arrow from cache (%s)", latest_cache_version.name)
        return Dataset.load_from_disk()
    elif latest_cache_version is not None and pickle_target.is_file():
        logger.info("Loading pickle from cache (%s)", latest_cache_version.name)
        with pickle_target.open('rb') as f:
            return dill.load(f)
    else:
        logger.info("Provided cache directory (%s) is empty. "
                    "Delete empty directory to check for older versions.",
                    latest_cache_version
                    )


def clear_cache(cache_path: str = '../data/NumTabQA/.cache',
                prefix: Optional[str] = None,
                postfix: Optional[str] = None,
                keep_latest: bool = True,
                force: bool = False,
                ) -> None:
    cache_path_obj = Path(cache_path)
    if not force and cache_path in ('', '.', './',):
        warnings.warn(f"Cache path {cache_path} is too general. This might delete the entire working directory! "
                      "If this is desired set force to True. Otherwise please specify different caching path."
                      )
        logger.info("Skip clearing cache.")
    elif cache_path_obj.exists():
        cache_versions = sorted(cache_path_obj.glob((prefix or '') + '*' + (postfix or '')))
        for v, version in enumerate(cache_versions):
            if keep_latest and v == (len(cache_versions) - 1):
                break
            if version.is_file():
                version.unlink()
            else:
                shutil.rmtree(version)
    else:
        warnings.warn(f"The provided cache_path '{cache_path}' does not exist! Nothing to delete.")
        logger.info("Skip clearing cache.")


def save_version(obj, cache_path, cache_file_name) -> None:
    save_path = Path(cache_path) / cache_file_name / datetime.now().strftime('%y%m%d_%H%M_%S_%f')
    logger.info(f"Writing {cache_file_name} to disk...")
    if (hasattr(obj, 'to_huggingface') and callable(obj.to_huggingface)):
        obj.to_huggingface().save_to_disk(save_path)
    elif isinstance(obj, list) and (hasattr(obj[0], 'to_state_dict') and callable(obj[0].to_state_dict)):
        Dataset.from_list([item.to_state_dict() for item in obj]).save_to_disk(save_path)
    else:
        warnings.warn("Saving with pickle is discouraged! "
                      "Consider implementing custom serialization (e.g to_huggingface, to_state_dict).")
        pickle_target = save_path / (cache_file_name + '.pickle')
        with pickle_target.open('wb') as f:
            dill.dump(f)


def delete_dataset(dataset) -> None:
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
