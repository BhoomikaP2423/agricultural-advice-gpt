import argparse
from pathlib import Path
import torch
from .model import MiniGPT, generate
from .tokenizer import BPETokenizer
from .train import DEFAULT_CHECKPOINT, DATA

def load_checkpoint(path=DEFAULT_CHECKPOINT):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Model checkpoint not found: {path}. Run: python -m agri_gpt train')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    tokenizer = BPETokenizer.from_dict(saved['tokenizer'])
    model = MiniGPT(**saved['config'])
    model.load_state_dict(saved['model']); model.eval()
    return model, tokenizer

def _choose_prompt(query, data_text):
    lines = [line.strip() for line in data_text.splitlines() if ':' in line]
    known = {line.split(':', 1)[0].lower() for line in lines}
    lowered = query.lower()
    crop_matches = [name for name in known if name != 'general' and name in lowered]
    crop = max(crop_matches, key=len).capitalize() if crop_matches else 'General'
    stop = {'with', 'from', 'that', 'this', 'have', 'your', 'what', 'when', 'does', 'show', 'help', 'crop', 'plant', 'plants'}
    terms = [w.strip('.,?!:;') for w in lowered.split() if len(w.strip('.,?!:;')) >= 4 and w not in stop and w not in crop.lower().split()]
    best_line, best_score = None, 0
    for line in lines:
        body = line.split(':', 1)[1].lower()
        score = sum(1 for term in terms if term in body)
        if score > best_score:
            best_line, best_score = line, score
    if best_line and best_score:
        body = best_line.split(':', 1)[1]
        lower_body = body.lower()
        positions = [lower_body.find(term) for term in terms if term in lower_body]
        prefix = best_line[:best_line.index(':') + 1] + body[:min(positions)]
        return best_line.split(':', 1)[0], prefix
    return crop, crop + ':'

def advise(query, checkpoint=DEFAULT_CHECKPOINT, max_new_tokens=180):
    model, tokenizer = load_checkpoint(checkpoint)
    data_text = Path(DATA).read_text(encoding='utf-8')
    crop, prompt = _choose_prompt(query, data_text)
    ids = tokenizer.encode(prompt)
    output = tokenizer.decode(generate(model, ids, max_new_tokens=max_new_tokens))
    answer = output[len(prompt):].split('\n')[0].strip()
    if len(answer.split()) < 8:
        candidates = [line.split(':', 1)[1].strip() for line in data_text.splitlines() if line.lower().startswith(crop.lower() + ':')]
        answer = candidates[0] if candidates else 'Monitor the crop regularly, maintain suitable soil moisture, and consult local agricultural extension guidance for a diagnosis.'
    return f'{crop}: {answer}'

def main():
    parser = argparse.ArgumentParser(prog='agri-gpt', description='Generate short crop-specific agricultural advice')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('advise', help='Generate advice for a crop or problem')
    p.add_argument('query', nargs='+', help='Crop or problem, e.g. tomato yellow leaves')
    p.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    p.add_argument('--tokens', type=int, default=180)
    t = sub.add_parser('train', help='Train the custom byte-level BPE GPT model')
    t.add_argument('--steps', type=int, default=500)
    t.add_argument('--data', default=str(DATA))
    t.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    if args.command == 'train':
        from .train import train
        train(args.data, args.checkpoint, args.steps)
    else:
        print(advise(' '.join(args.query), args.checkpoint, args.tokens))

if __name__ == '__main__':
    main()
