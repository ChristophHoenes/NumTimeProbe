from functools import partial

import datasets
from torch.utils.data.dataloader import DataLoader

from numerical_table_questions.data_loading import TableQADataModule
from numerical_table_questions.lazy_data_processing import QuestionTableIndexDataset


#dat = datasets.Dataset.load_from_disk('/home/mamba/.cache/gittables_group_filtered_standard_templates_test_100k/241016_1417_12_123456')
dat = datasets.Dataset.load_from_disk('/home/mamba/.cache/16_few_shot_tables_from_100k_test_set')
dataset = QuestionTableIndexDataset(dat)
batch_size = 16
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x: x if batch_size is not None and batch_size > 1 else None)

dataloader_iterator = iter(dataloader)
batch = next(dataloader_iterator)

for i, b in enumerate(dataloader):
    print(b['question_id'])
    print(i)
