from agri_gpt.tokenizer import BPETokenizer

def test_round_trip_ascii():
    t = BPETokenizer.train('tomato plant\ntomato fruit', 20)
    assert t.decode(t.encode('tomato plant')) == 'tomato plant'

def test_round_trip_unicode():
    t = BPETokenizer.train('soil – water\nsoil health', 20)
    assert t.decode(t.encode('soil – water')) == 'soil – water'

def test_vocab_size():
    t = BPETokenizer.train('abababab\nabababab', 5)
    assert t.vocab_size >= 256

def test_empty_text():
    t = BPETokenizer.train('', 10)
    assert t.encode('') == []
    assert t.decode([]) == ''

def test_serialization():
    t = BPETokenizer.train('crop crop crop', 5)
    restored = BPETokenizer.from_dict(t.to_dict())
    assert restored.decode(restored.encode('crop')) == 'crop'

def test_ids_in_vocab():
    t = BPETokenizer.train('irrigation irrigation', 8)
    assert all(0 <= i < t.vocab_size for i in t.encode('irrigation'))
