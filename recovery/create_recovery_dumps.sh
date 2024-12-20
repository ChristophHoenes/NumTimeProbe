# This will run some binary string searches on the file system to recover accidentally deleted code snippets

# general dump (search for project specific import)
grep -a -B 20 -A 1000 -F 'from numerical_table_questions.' /dev/mapper/ubuntu--vg-home > /scratch/choenes/manual_recovery/source_code_dump.txt

# file specific targeted search queries
# sql_templates_recovery
grep -a -B 20 -A 500 -F "OP = TypeVar('OP', bound='SQLOperatorTemplate')" /dev/mapper/ubuntu--vg-home > /scratch/choenes/manual_recovery/sql_templates_recovery.txt

# table_recovery
grep -a -B 30 -A 500 -F "# NUMBER_REGEX = re.compile(r'(\d(,\d{3})*|\d+)?(\.\d+)?') # old expression with no negative" /dev/mapper/ubuntu--vg-home > /scratch/choenes/manual_recovery/table_recovery.txt

# data_caching_recovery
grep -a -B 200 -A 150 -F '# timed cleanup with threads proved impractical and was solved differently with cached propery and weak reference' /dev/mapper/ubuntu--vg-home > /scratch/choenes/manual_recovery/data_caching_recovery.txt

# arguments_recovery
grep -a -B 400 -A 100 -F 'default="./data/NumTabQA/.cache", help="File path to the location in the file system where the data is stored."' /dev/mapper/ubuntu--vg-home > /scratch/choenes/manual_recovery/arguments_recovery.txt

# sql_util_recovery
grep -a -B 20 -A 100 -F 'from pandasql import sqldf # TODO try duckdb as drop-in replacement' /dev/mapper/ubuntu--vg-home > /scratch/choenes/manual_recovery/sql_util_recovery.txt

# table_recovery_2
grep -a -B 350 -A 150 -F '"Either a non-empty extension_string or use_numbering=True must be used!"' /dev/mapper/ubuntu--vg-home > /scratch/choenes/manual_recovery/table_recovery_2.txt

# sql_util_recovery_2
grep -a -B 70 -A 100 -F 'def execute_sql(query: str,' /dev/mapper/ubuntu--vg-home > /scratch/choenes/manual_recovery/sql_util_recovery_2.txt
