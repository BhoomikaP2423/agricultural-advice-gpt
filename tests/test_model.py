import pytest, torch
from agri_gpt.model import MiniGPT

def make_model(): return MiniGPT(vocab_size=270, context=16, width=32, heads=4, layers=2, dropout=0)

def test_output_shape():
    m=make_model(); logits, loss=m(torch.randint(0,270,(2,8)))
    assert logits.shape == (2,8,270) and loss is None

def test_loss_is_scalar():
    m=make_model(); x=torch.randint(0,270,(2,8)); _,loss=m(x,x)
    assert loss.ndim == 0 and torch.isfinite(loss)

def test_context_limit():
    with pytest.raises(ValueError): make_model()(torch.randint(0,270,(1,17)))

def test_batch_one():
    logits,_=make_model()(torch.randint(0,270,(1,4)))
    assert logits.shape[0] == 1

def test_different_sequence_lengths():
    m=make_model()
    assert m(torch.randint(0,270,(1,3)))[0].shape[1] == 3
    assert m(torch.randint(0,270,(1,7)))[0].shape[1] == 7

def test_parameters_are_trainable():
    assert any(p.requires_grad for p in make_model().parameters())

def test_training_step_updates_weights():
    m=make_model(); x=torch.randint(0,270,(2,8)); before=m.head.weight.detach().clone()
    _,loss=m(x,x); loss.backward(); torch.optim.Adam(m.parameters()).step()
    assert not torch.equal(before,m.head.weight.detach())
