import datasets

dataset_size = 61
dataset = datasets.Dataset.from_list([{'idx': i} for i in range(dataset_size)])


def test_func(row, idx):
    if idx == 58:
        return {'output': []}
    else:
        return {'output' : [{'test': 1}, {'test': 2}]}


# this works fine
test1 = dataset.map(lambda row, idx: test_func(row, idx), with_indices=True, num_proc=4)

# this fails
#test2 = dataset.map(lambda row, idx: test_func(row, idx), with_indices=True, num_proc=32)

# this should work
#features = {'output': datasets.Sequence(feature=datasets.Value(dtype='null', id=None), length=-1, id=None)}
#features.update(dataset.features)
#features = {'idx': {'dtype': 'int64', 'id': None, '_type': 'Value'},
#            'output': {'_type': 'Sequence', 'feature': {'dtype': 'struct<test: int64>', 'id': None, '_type': 'Value'}, 'length': -1, 'id': None}
#            }
features = dataset.features
features["output"] = [{"test": datasets.Value("int64")}]
test2 = dataset.map(lambda row, idx: test_func(row, idx), with_indices=True, num_proc=32, features=features)  # features=datasets.Features.from_dict(features))


dataset_small = datasets.Dataset.from_list([{'idx': i} for i in range(10)])
test3 = dataset_small.map(lambda row, idx: test_func(row, idx), with_indices=True, num_proc=32, features=features)

test4 = datasets.concatenate_datasets([dataset, dataset_small])
