from typing import List, Optional

import seqio
import tensorflow as tf
from transformers import PreTrainedTokenizerBase


class HFVocabulary(seqio.Vocabulary):
    """A Vocabulary implementation that uses a HuggingFace tokenizer."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, use_eos_as_pad: bool = False
    ):
        self.tokenizer = tokenizer
        self.use_eos_as_pad = use_eos_as_pad

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self) -> int:
        if self.use_eos_as_pad:
            id = self.eos_id()
        else:
            id = self.tokenizer.pad_token_id
        if id is None:
            raise ValueError("Tokenizer does not have a pad token")
        return id

    def unk_id(self) -> Optional[int]:
        return self.tokenizer.unk_token_id

    @property
    def extra_ids(self) -> int:
        # hf tokenizer manages its own special tokens
        return 0

    def __len__(self):
        return len(self.tokenizer)

    def __eq__(self, other):
        return isinstance(other, HFVocabulary) and self.tokenizer == other.tokenizer

    def _base_vocab_size(self) -> int:
        return len(self.tokenizer)

    def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        s = s.numpy().decode("utf-8")
        return self.tokenizer.encode(s, add_special_tokens=False, return_tensors="tf")

    def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        return self.tokenizer.decode(ids)
