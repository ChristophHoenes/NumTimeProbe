from copy import deepcopy

import pytest
import torch

from numerical_table_questions.model import order_positional_arguments


INPUTS = [torch.rand([10, 512]), torch.ones([10, 512])*2, torch.zeros([10, 512])]
TARGETS = torch.ones([10, 512])


def test_side_effects():
    # define inputs
    inputs = [tensor.clone() for tensor in INPUTS]
    targets = TARGETS.clone()
    input_map_in = {'*': None, 'some_kwarg': 123}
    initial_len_input_map = len(input_map_in)
    # apply function on inputs
    order_positional_arguments(inputs, targets, input_map=input_map_in)
    # test if inputs changed through execution
    assert '*' not in input_map_in, "After calling order_positional_arguments the '*' syntax sugar should be computed explicitly and disappear from input_map!"
    assert all([i in input_map_in for i in range(len(inputs))]), "All elements of 'inputs' should occur with their position mapping in input_map after function was called!"
    assert input_map_in.get('some_kwarg') == 123, f"All additional keyword arguments should remain unchanged, but 'some_kwarg' is f{input_map_in.get('some_kwarg')} instead of 123!"
    # keyword (str key) args stay the same, only '*' should be gone and len(inputs) keys should be added. No keys besides those should be added
    assert len(input_map_in) == len(inputs) + initial_len_input_map - 1, f"The number of keys in input_map should be exactly the number of inputs, but are {len(input_map_in)} vs. {len(inputs)}!"
    # inputs and targets should still be the same
    assert all([torch.all(tensor.clone() == inputs[i]).item() for i, tensor in enumerate(INPUTS)]), "Input argument inputs has changed, but should be free of side effects!"
    assert torch.all(TARGETS.clone() == targets), "Input argument targets has changed, but should be free of side effects!"
    # repeated calls after the first one should be side completely effect free
    input_map_copy = deepcopy(input_map_in)
    order_positional_arguments(inputs, targets, input_map=input_map_in)
    assert all([k1 == k2 and v1 == v2
                for (k1, v1), (k2, v2) in zip(input_map_copy.items(), input_map_in.items())
                ]
               ), "Input_map changed after second call of order_positional_arguments but should stay the same!"


def test_injection():
    inputs = [tensor.clone() for tensor in INPUTS]
    targets = TARGETS.clone()
    # inject any constant
    input_map1 = {'*': None, 1: lambda x, y: 'some_text', 'some_kwarg': 123}
    # ideas for more tests (passing subset of inputs currently not supported)
    # input_map2 = {'*': None, 0: lambda x, y: x[1], 1: lambda x, y: x[0], 'some_kwarg': 123}
    # input_map3 = {1: lambda x, y: 0, 'some_kwarg': 123}
    # input_map4 = {1: lambda x, y: x[1] + 1, 'some_kwarg': 123}
    results = order_positional_arguments(inputs, targets, input_map=input_map1)
    assert len(results) + 1 == len(input_map1), "The length of input_map should reflect the length of the output plus the keyword arguments!"
    inputs.insert(1, 'some_text')
    assert all([torch.all(torch.tensor(tensor == results[i])).item() for i, tensor in enumerate(inputs)]), "The specified argument has not been properly injected!"
    del inputs[1]
    # inject targets in input
    input_map2 = {'*': None, 1: lambda x, y: y, 'some_kwarg': 123}
    results = order_positional_arguments(inputs, targets, input_map=input_map2)
    inputs.insert(1, targets)
    assert all([torch.all(tensor == results[i]).item() for i, tensor in enumerate(inputs)]), "The specified argument has not been properly injected!"
    del inputs[1]
    # inject any input again
    input_map3 = {'*': None, 1: 0, 'some_kwarg': 123}
    results = order_positional_arguments(inputs, targets, input_map=input_map3)
    inputs.insert(1, inputs[0])
    assert all([torch.all(tensor == results[i]).item() for i, tensor in enumerate(inputs)]), "The specified argument has not been properly injected!"


def test_selection():
    inputs = [tensor.clone() for tensor in INPUTS]
    targets = TARGETS.clone()
    # swap order of inputs
    input_map1 = {1: 0, 0: 1, 2: 2, 'some_kwarg': 123}
    results = order_positional_arguments(inputs, targets, input_map=input_map1)
    expected = [inputs[1], inputs[0], inputs[2]]
    assert all([torch.all(tensor == results[i]).item() for i, tensor in enumerate(expected)]), "The specified arguments have not been properly swapped!"
    # select subset of input
    input_map2 = {0: 0, 1: 2, 'some_kwarg': 123}
    results = order_positional_arguments(inputs, targets, input_map=input_map2)
    expected = [inputs[0], inputs[2]]
    assert all([torch.all(tensor == results[i]).item() for i, tensor in enumerate(expected)]), "The specified subset has not been properly selected!"
