"""A compact decoder-only GPT with token/position embeddings and causal attention."""
import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width), nn.Dropout(dropout))

    def forward(self, x):
        length = x.size(1)
        mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        y = self.norm1(x)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.norm2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, context=96, width=96, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*[TransformerBlock(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):
        batch, length = idx.shape
        if length > self.context:
            raise ValueError(f"Sequence length {length} exceeds context {self.context}")
        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]
        logits = self.head(self.norm(self.blocks(x)))
        loss = None if targets is None else nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

@torch.no_grad()
def generate(model, token_ids, max_new_tokens=180, temperature=0.8, top_k=24):
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([token_ids[-model.context:]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.context:])
        scores = logits[:, -1, :] / max(temperature, 1e-4)
        k = min(top_k, scores.size(-1))
        values, _ = torch.topk(scores, k)
        scores[scores < values[:, [-1]]] = -float('inf')
        probs = torch.softmax(scores, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat((idx, next_id), dim=1)
        if next_id.item() == 10:
            break
    return idx[0].tolist()
