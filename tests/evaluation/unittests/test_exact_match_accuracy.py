import pandas as pd
import pytest
import torch

from numerical_table_questions.evaluation import exact_match_accuracy
from numerical_table_questions.tokenizer_utils import get_tokenizer


DEFAULT_TOKENIZER = get_tokenizer('tapex')


def test_correct_example(tokenizer=DEFAULT_TOKENIZER, batch_size=3, sequence_lenth=512, vocab_size=2_137, target_offset=179, mask_token_id=-100):
    # make sure target offset does not exceed sequence length
    if sequence_lenth < (target_offset + 20):
        target_offset = max([0, sequence_lenth-21])

    string_answer = ' result: -1.25 '
    # dummy_table = pd.DataFrame.from_dict({'dummy_table_col': ["1"]})
    answer_tokens = tokenizer(
        # table=[dummy_table, dummy_table],
        # query=['dummy frage1?', 'wer wie wo was ist die frage nummer 2?'],
        # answer=[string_answer, string_answer + '?? .. werwe? werw rw werwr'],
        answer=string_answer,
        add_special_tokens=False,
    )['input_ids']
    sample_target = torch.ones([1, sequence_lenth]).long() * mask_token_id
    sample_target[:, target_offset:target_offset+len(answer_tokens)] = torch.tensor(answer_tokens).long()
    sample_target = sample_target.repeat(batch_size, 1)

    # make sure logits at target sequence positions have the highest value at the answer token ID
    # such that softmax predicts the correct answer tokens
    dummy_logits = torch.rand(batch_size, sequence_lenth, vocab_size)
    max_value_vocab_logits = dummy_logits.max()
    for sample in dummy_logits:
        for i, token_id in enumerate(answer_tokens):
            sample[target_offset+i, token_id] = max_value_vocab_logits + torch.rand(1).abs()
    model_outputs = {'loss': 0.732,
                     'logits': dummy_logits,
                     'attention_something': torch.ones(3, 512),
                     }

    metric_value, eval_results = exact_match_accuracy(model_outputs, sample_target, tokenizer)
    assert metric_value == 1.0, f"Only correct examples should result in exact_match of 1.0 but was {metric_value}!"
    assert eval_results == [True] * batch_size, "Only correct examples should result in a list of only true values in eval_results!"
