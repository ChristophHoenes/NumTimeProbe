import logging
import logging.config
from pathlib import PurePath
from typing import Optional, Union

import pandas as pd
from pandasql import sqldf


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
        except Exception as e:
            logger.info('query:\n', query)
            logger.info('table:\n', dataframe.head(5))
            raise e
    if query_result is None and debug_empty_result:
        logger.debug('query:\n', query)
        logger.debug('table:\n', dataframe.head(3))
    return query_result
