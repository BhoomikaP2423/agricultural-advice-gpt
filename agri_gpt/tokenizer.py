"""Small byte-level BPE tokenizer trained from the supplied advice corpus."""
from collections import Counter

class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = [tuple(pair) for pair in (merges or [])]
        self._rebuild()

    def _rebuild(self):
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.token_bytes = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.token_bytes[256 + i] = self.token_bytes[a] + self.token_bytes[b]

    @classmethod
    def train(cls, text, num_merges=96):
        sequences = [list(line.encode('utf-8')) for line in text.splitlines() if line.strip()]
        merges = []
        for _ in range(num_merges):
            counts = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < 2:
                break
            merges.append(pair)
            updated = []
            for seq in sequences:
                out, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                        out.append(256 + len(merges) - 1)
                        i += 2
                    else:
                        out.append(seq[i]); i += 1
                updated.append(out)
            sequences = updated
        return cls(merges)

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        for rank, pair in enumerate(self.merges):
            merged, i = [], 0
            new_id = 256 + rank
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                    merged.append(new_id); i += 2
                else:
                    merged.append(ids[i]); i += 1
            ids = merged
        return ids

    def decode(self, ids):
        return b''.join(self.token_bytes[int(i)] for i in ids).decode('utf-8', errors='replace')

    def to_dict(self):
        return {'merges': [list(p) for p in self.merges]}

    @classmethod
    def from_dict(cls, data):
        return cls(data['merges'])
