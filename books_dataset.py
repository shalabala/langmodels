import struct
import torch
from torch.utils.data import Dataset

class BooksDataset(Dataset):
    def __init__(self, sequence_len: int, tokens_bin_file: str, vocab_file: str):
        self.sequence_len = sequence_len
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

        self.tokens = tokens
        self.vocab = {0: '<OOV>'}  # adding the out of vocabulary token
        with open(vocab_file) as vf:
            for line in vf:
                last_colon_idx = line.rfind(':')
                self.vocab[int(line[last_colon_idx+1:-1])
                           ] = line[:last_colon_idx]

    def __len__(self):
        return self.tokens.shape[0] - self.sequence_len -1 # this is so that we always give back data points of the same length

    def __getitem__(self, i):
        return (self.tokens[i: i+self.sequence_len], self.tokens[i+1: i+self.sequence_len+1])
        
