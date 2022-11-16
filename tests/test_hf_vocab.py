import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

from hfseqio import HFVocabulary

tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab = HFVocabulary(tokenizer, use_eos_as_pad=True)

TEST_STRING = "this is a test"
UNK_STRING = " ⁇ "
TEST_TOKENS = tuple(tokenizer.encode(TEST_STRING, add_special_tokens=False))


def test_vocab():
    assert len(tokenizer) == vocab.vocab_size
    assert TEST_TOKENS == tuple(vocab.encode(TEST_STRING))
    assert TEST_STRING == vocab.decode(TEST_TOKENS)
    assert TEST_TOKENS == tuple(vocab.encode_tf(tf.constant(TEST_STRING)).numpy().tolist())
    assert TEST_STRING == vocab.decode_tf(tf.constant(TEST_TOKENS))
