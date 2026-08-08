import json
import os

class MathTokenizer:
    def __init__(self):
        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4

    def encode(self, text):
        return [1] + [self.char2idx.get(c, 3) for c in text] + [2]

    def decode(self, indices):
        return "".join([self.idx2char.get(int(idx), '<UNK>') for idx in indices if idx not in [0, 1, 2]])

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.char2idx = data['char2idx']
            self.idx2char = {int(k): v for k, v in data['idx2char'].items()}
            self.vocab_size = len(self.char2idx)