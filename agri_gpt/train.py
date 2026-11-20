import argparse, json, random
from pathlib import Path
import torch
from .tokenizer import BPETokenizer
from .model import MiniGPT

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'advice.txt'
DEFAULT_CHECKPOINT = ROOT / 'checkpoints' / 'agri_gpt.pt'

def train(data_path=DATA, checkpoint=DEFAULT_CHECKPOINT, steps=500, num_merges=96, seed=17):
    random.seed(seed); torch.manual_seed(seed)
    text = Path(data_path).read_text(encoding='utf-8')
    tokenizer = BPETokenizer.train(text, num_merges=num_merges)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    context, width, heads, layers = 96, 96, 4, 2
    model = MiniGPT(tokenizer.vocab_size, context, width, heads, layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    if len(ids) <= context + 1:
        raise ValueError('Training corpus is too small for the selected context length')
    model.train()
    for step in range(steps):
        starts = torch.randint(0, len(ids) - context - 1, (16,))
        x = torch.stack([ids[s:s+context] for s in starts])
        y = torch.stack([ids[s+1:s+context+1] for s in starts])
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % max(1, steps // 5) == 0:
            print(f'Step {step+1}/{steps} - loss: {loss.item():.4f}')
    checkpoint = Path(checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'tokenizer': tokenizer.to_dict(), 'config': {'vocab_size': tokenizer.vocab_size, 'context': context, 'width': width, 'heads': heads, 'layers': layers}}, checkpoint)
    print(f'Saved model to {checkpoint}')
    return checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the agricultural advice mini-GPT')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--data', default=str(DATA))
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    train(args.data, args.checkpoint, args.steps)
