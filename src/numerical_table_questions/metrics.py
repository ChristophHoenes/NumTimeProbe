from typing import List, Literal, Optional
from time import time

import torch


def accuracy_extract_target_position(model_outputs, ground_truth, compare_strings=False, **kwargs):
    if kwargs.get('tokenizer') is None:
        raise ValueError("Expected 'tokenizer' to be passed in metric's kwargs, but argument was not found!")

    # determine masked positions (targets) in sequence
    target_positions = torch.nonzero(ground_truth != -100)

    # convert logits to token_ids
    predicted_tokens = dict()
    ground_truth_tokens = dict()
    for pos in target_positions:
        if pos[0] not in predicted_tokens:
            predicted_tokens[pos[0].item()] = [model_outputs['logits'][pos[0], pos[1]].softmax(dim=-1).argmax(dim=-1).item()]
            ground_truth_tokens[pos[0].item()] = [ground_truth[pos[0], pos[1]].item()]
        else:
            predicted_tokens[pos[0].item()].append(model_outputs['logits'][pos[0], pos[1]].softmax(dim=-1).argmax(dim=-1).item())
            ground_truth_tokens[pos[0].item()].append(ground_truth[pos[0], pos[1]].item())
    if compare_strings:
        # decode token ids into strings
        tokenizer = kwargs['tokenizer']
        target_tokens_without_mask = []
        for i in range(len(ground_truth)):
            target_tokens_without_mask.append([ground_truth[pos[0], pos[1]] for pos in target_positions if pos[0] == i])
        string_targets = [''.join(tokenizer.decode(tokens, skip_special_tokens=True)) for tokens in target_tokens_without_mask]
        #print('string_targets: ', string_targets)

        # probably target it is unnecessary to sort
        # sorted([(idx, tokenizer.batch_decode(tokens, skip_special_tokens=True)) for idx, tokens in predicted_tokens.items()])
        predictions = tokenizer.batch_decode(list(predicted_tokens.values()), skip_special_tokens=True)
        #predictions = [''.join(tokenizer.decode(tokens, skip_special_tokens=True)) for tokens in predicted_tokens.values()]
        #print('predictions: ', predictions)
        #print('batch_predictions: ', predictions2)
        eval_results = [sample[0].strip() == sample[1].strip()
                        for sample in zip(predictions, string_targets)]
    else:
        #eval_results = ground_truth[target_positions] == torch.vstack([torch.tensor(token_list) for token_list in predicted_tokens.values()])
        eval_results = [sample[0] == sample[1]
                        for sample in zip(ground_truth_tokens.values(), predicted_tokens.values())]
    return sum(eval_results) / len(eval_results), eval_results


def str_match_accuracy(predictions, targets):
    is_correct = [pred == targ for pred, targ in zip(predictions, targets)]
    return sum(is_correct)/len(is_correct), is_correct


def float_match_accuracy(predictions, targets, tolerance=0.0):
    is_correct = []
    for pred, target in zip(predictions, targets):
        try:
            if tolerance > 0:
                is_correct.append(abs(float(target) - float(pred)) <= tolerance)
            else:
                is_correct.append(float(target) == float(pred))
        except ValueError:
            is_correct.append(False)
    return sum(is_correct)/len(is_correct), is_correct


def absolute_distance(predictions: List[str], targets: List[str], aggregation: Literal['mean', 'median'] = 'median', in_percent=True, nan_distance: Optional[float] = None):
    target_values = [float(target) if target.lower() not in ['none', '', 'nan'] else float('-inf') for target in targets]
    prediction_values = []
    for pred in predictions:
        try:
            prediction_values.append(float(pred))
        except ValueError:
            prediction_values.append(None)
    # what distance/penalty to use for nan values in predictions
    if nan_distance is None:
        target_magnitude = [abs(v) for v in target_values]
        nan_distance = sum(target_magnitude)/len(target_magnitude) if aggregation == 'mean' else sorted(target_values)[len(target_values)//2]
    distances = []
    for pred, target_value in zip(prediction_values, target_values):
        if in_percent:
            if pred is None:
                distances.append(10.0)
            else:
                distances.append(min(10.0, abs(target_value - pred) / abs(target_value + 1e-7)))
        else:
            if pred is None:
                distances.append(nan_distance)
            else:
                distances.append(abs(target_value - float(pred)))
    metric_result = sum(distances)/len(distances) if aggregation == 'mean' else sorted(distances)[len(distances)//2]
    return metric_result, distances


def token_accuracy(model_outputs, ground_truth):
    # token_ids = [model.tokenize(gt) for gt in ground_truth]
    raise NotImplementedError
    # TODO implement
    eval_results = [sample[0] == sample[1]
                    for sample in zip(predictions, token_ids)]
    return sum(eval_results) / len(eval_results), eval_results
