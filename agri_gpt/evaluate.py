"""Report held-out next-token loss and perplexity for the trained language model."""
from pathlib import Path
import math, torch
from .tokenizer import BPETokenizer
from .model import MiniGPT
from .train import DATA, DEFAULT_CHECKPOINT

def evaluate(data_path=DATA, checkpoint=DEFAULT_CHECKPOINT):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    tokenizer = BPETokenizer.from_dict(saved['tokenizer'])
    model = MiniGPT(**saved['config']); model.load_state_dict(saved['model']); model.eval()
    text = Path(data_path).read_text(encoding='utf-8')
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = max(1, int(len(ids) * 0.9))
    validation = ids[split:]
    if len(validation) < 2:
        raise ValueError('Not enough held-out tokens to evaluate')
    losses = []
    with torch.no_grad():
        for start in range(0, len(validation) - 1, model.context):
            chunk = validation[start:start + model.context + 1]
            if len(chunk) < 2: continue
            _, loss = model(chunk[:-1][None, :], chunk[1:][None, :])
            losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return {'validation_loss': round(mean_loss, 4), 'perplexity': round(math.exp(min(mean_loss, 20)), 3), 'validation_tokens': len(validation)}

if __name__ == '__main__':
    print(evaluate())
