import logging
import logging.config
from pathlib import PurePath
from typing import Optional, Union

import pandas as pd
from pandasql import sqldf  # TODO try duckdb as drop-in replacement


log_file_init_path = str(PurePath(__file__).parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def execute_sql(query: str, dataframe: pd.DataFrame, debug_empty_result=False,
                ) -> Optional[Union[pd.Series, pd.DataFrame]]:
    if query is None:
        raise ValueError("Can only compute the answer to a question if the \
                            corresponding sql_query to answer is available!"
                         )
    else:
        df = dataframe  # renaming for sqldf to find table
        try:
            query_result = sqldf(query)
        except Exception as e:  # Log issue with query but do not interrupt program -> return None instead
            logger.warn("Query execution failed with exception! "
                        "See logging level debug for more info on query and table and level critical for the stack trace."
                        )
            logger.critical(e, exc_info=True)
            logger.info('query:\n', query)
            logger.info('table:\n', dataframe.head(5))
            query_result = None
    if query_result is None and debug_empty_result:
        logger.debug('query:\n', query)
        logger.debug('table:\n', dataframe.head(3))
    return query_result
