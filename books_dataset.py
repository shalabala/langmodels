import struct
import torch
from torch.utils.data import Dataset


class BooksDataset(Dataset):
    def __init__(self,
                 sequence_len: int,
                 tokens: torch.Tensor,
                 vocab: dict[int, str],
                 len: int,
                 offset: int = 0):
        self.sequence_len = sequence_len
        self.tokens = tokens
        self.vocab = vocab
        self.offset = offset
        self.len = len

    @staticmethod
    def from_file(sequence_len: int, tokens_bin_file: str, vocab_file: str):
        with open(tokens_bin_file, "rb") as tf:
            count = struct.unpack("<Q", tf.read(8))[0]
            i = 0
            batch = 4096 * 20000  # around 100mb, batching reduces unnecessary mem usage
            tokens = torch.asarray([], dtype=torch.uint16)
            while i < count:
                shorts_to_unpack = min(batch, count - i)
                tokens_py = struct.unpack(
                    f"<{shorts_to_unpack}H", tf.read(2*shorts_to_unpack))
                tokens = torch.cat((tokens,  torch.asarray(
                    tokens_py, dtype=torch.uint16)))
                del tokens_py
                i += shorts_to_unpack

        vocab = {0: '<OOV>'}  # adding the out of vocabulary token
        with open(vocab_file) as vf:
            for line in vf:
                last_colon_idx = line.rfind(':')
                vocab[int(line[last_colon_idx+1:-1])
                      ] = line[:last_colon_idx]

        # this is so that we always give back data points of the same length
        len = tokens.shape[0] - sequence_len - 1
        return BooksDataset(sequence_len, tokens, vocab, len)

    def __len__(self):
        return self.len

    def split_off(self, no_of_items):
        if (no_of_items > self.len):
            raise ValueError(
                "When splitting off a chunk of the dataset the number of elements cannot exceed the length of the original dataset!")
        split_off_dataset = BooksDataset(
            self.sequence_len, self.tokens, self.vocab, no_of_items, self.len - no_of_items)
        self.len -= no_of_items
        return split_off_dataset

    def unsplit(self):
        self.len = self.tokens.shape[0] - self.sequence_len - 1

    def __getitem__(self, i):
        return (self.tokens[i+self.offset: i+self.offset+self.sequence_len],
                self.tokens[i+self.offset+1: i+self.offset+self.sequence_len+1])
