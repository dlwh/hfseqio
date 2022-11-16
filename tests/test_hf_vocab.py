import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

from hfseqio import HFVocabulary

tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab = HFVocabulary(tokenizer)

TEST_STRING = "this is a test"
UNK_STRING = " ⁇ "
TEST_TOKENS = tokenizer.encode(TEST_STRING, add_special_tokens=False)


def _decode_tf(vocab, tokens):
    results = vocab.decode_tf(tf.constant(tokens, tf.int32)).numpy()

    def _apply(fun, sequence):
        if isinstance(sequence, (list, np.ndarray)):
            return [_apply(fun, x) for x in sequence]
        else:
            return fun(sequence)

    return _apply(lambda x: x.decode("UTF-8"), results)


def test_decode_tf():
    for rank in range(0, 3):
        ids = TEST_TOKENS
        expected_str = TEST_STRING

        # Creates an arbitrarly nested tensor.
        for _ in range(rank):
            ids = [ids]
            expected_str = [expected_str]

        # single sequences to decode
        assert expected_str == _decode_tf(vocab, ids)

        # multiple sequences to decode
        res = _decode_tf(vocab, (ids, ids))
        exp = [expected_str] * 2
        assert exp == res


def test_vocab():
    assert len(tokenizer) == vocab.vocab_size
    assert TEST_TOKENS == vocab.encode(TEST_STRING)
    assert TEST_STRING == vocab.decode(TEST_TOKENS)
    assert TEST_TOKENS == tuple(vocab.encode_tf(tf.constant(TEST_STRING)).numpy())
    assert TEST_STRING == _decode_tf(vocab, tf.constant(TEST_TOKENS))
