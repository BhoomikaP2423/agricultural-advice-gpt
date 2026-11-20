from pathlib import Path
import torch
from agri_gpt.tokenizer import BPETokenizer
from agri_gpt.model import MiniGPT
from agri_gpt.evaluate import evaluate

def test_evaluation_returns_metrics(tmp_path):
    data=tmp_path/'data.txt'; data.write_text(('Tomato: water plants carefully.\n'*20),encoding='utf-8')
    tok=BPETokenizer.train(data.read_text(),8)
    config={'vocab_size':tok.vocab_size,'context':16,'width':32,'heads':4,'layers':1}
    model=MiniGPT(**config); checkpoint=tmp_path/'model.pt'
    torch.save({'model':model.state_dict(),'tokenizer':tok.to_dict(),'config':config},checkpoint)
    result=evaluate(data,checkpoint)
    assert {'validation_loss','perplexity','validation_tokens'} <= result.keys()
    assert result['perplexity'] > 0
