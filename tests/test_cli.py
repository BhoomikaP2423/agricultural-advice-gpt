from pathlib import Path
import pytest, torch
from agri_gpt.tokenizer import BPETokenizer
from agri_gpt.model import MiniGPT
from agri_gpt.cli import load_checkpoint, advise

def test_missing_checkpoint_message(tmp_path):
    with pytest.raises(FileNotFoundError, match='Run: python -m agri_gpt train'):
        load_checkpoint(tmp_path/'missing.pt')

def test_load_saved_checkpoint(tmp_path):
    tok=BPETokenizer.train('Tomato: water plants', 5)
    config={'vocab_size':tok.vocab_size,'context':16,'width':32,'heads':4,'layers':1}
    model=MiniGPT(**config); path=tmp_path/'model.pt'
    torch.save({'model':model.state_dict(),'tokenizer':tok.to_dict(),'config':config},path)
    loaded, loaded_tok=load_checkpoint(path)
    assert isinstance(loaded,MiniGPT) and loaded_tok.vocab_size==tok.vocab_size

def test_advice_fallback_for_short_generation(tmp_path):
    tok=BPETokenizer.train('Tomato: water plants deeply in the morning.', 5)
    config={'vocab_size':tok.vocab_size,'context':96,'width':32,'heads':4,'layers':1}
    model=MiniGPT(**config)
    path=tmp_path/'model.pt'
    torch.save({'model':model.state_dict(),'tokenizer':tok.to_dict(),'config':config},path)
    result=advise('tomato yellow leaves',path,max_new_tokens=1)
    assert result.startswith('Tomato:') and len(result.split()) > 5

def test_empty_query_uses_general(tmp_path):
    tok=BPETokenizer.train('General: monitor crops regularly and check soil moisture.', 5)
    config={'vocab_size':tok.vocab_size,'context':96,'width':32,'heads':4,'layers':1}
    model=MiniGPT(**config); path=tmp_path/'model.pt'
    torch.save({'model':model.state_dict(),'tokenizer':tok.to_dict(),'config':config},path)
    assert advise('',path,max_new_tokens=1).startswith('General:')

def test_problem_query_selects_relevant_training_prefix():
    from agri_gpt.cli import _choose_prompt
    crop, prompt = _choose_prompt('fall armyworm damage', 'Maize: For fall armyworm, scout whorls regularly.\nTomato: Water deeply.')
    assert crop == 'Maize'
    assert prompt == 'Maize: For '

def test_crop_query_selects_crop_prefix():
    from agri_gpt.cli import _choose_prompt
    crop, prompt = _choose_prompt('tomato', 'Tomato: Water deeply.\nRice: Monitor pests.')
    assert crop == 'Tomato' and prompt == 'Tomato:'
