# Agricultural Advice GPT

A small command-line agricultural advice generator. It accepts a crop name or a problem description and generates a concise advice snippet using a custom-trained, decoder-only GPT language model.

## What it demonstrates

- **Byte-pair tokenization:** a byte-level BPE tokenizer learns frequent adjacent byte merges from the supplied farming-advice corpus.
- **Embeddings:** learned token embeddings and positional embeddings represent the input sequence.
- **Causal self-attention:** masked multi-head self-attention prevents a token from attending to future tokens.
- **Transformer blocks:** pre-normalized attention and feed-forward layers form a compact GPT-style decoder.
- **Autoregressive generation:** the model predicts one token at a time from a crop prompt.
- **CLI:** train the model and request crop-specific advice from a terminal.

This is an educational mini-GPT, not a production agricultural decision system. The included corpus is deliberately small and general. Always verify advice with local agricultural extension services, soil tests, product labels, and region-specific guidance. Do not treat generated pesticide or disease advice as a diagnosis.

## Project structure

```text
agricultural-advice-gpt/
├── agri_gpt/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── evaluate.py
│   ├── model.py
│   ├── tokenizer.py
│   └── train.py
├── data/advice.txt
├── tests/
├── .github/workflows/ci.yml
├── RESULTS.md
├── requirements.txt
└── README.md
```

## Run locally or in Google Colab

Python 3.10+ is recommended. A CPU works for this small model; Google Colab GPU will make training faster.

```bash
pip install -r requirements.txt
python -m pytest -q
python -m agri_gpt train --steps 500
python -m agri_gpt advise tomato
python -m agri_gpt advise "tomato yellow leaves"
python -m agri_gpt advise "maize fall armyworm"
python -m agri_gpt advise "wheat rust"
python -m agri_gpt.evaluate
```

The first training run creates `checkpoints/agri_gpt.pt`. The checkpoint is ignored by Git because it can be regenerated from the source corpus. To train for longer, increase `--steps` (for example, 1000). Training time depends on the machine.

## How it works

1. `data/advice.txt` contains short crop-specific and general farming advice snippets.
2. `BPETokenizer.train()` learns byte-pair merges from the corpus and encodes text as token IDs.
3. `MiniGPT` combines token and position embeddings, causal multi-head self-attention, feed-forward layers, and a vocabulary projection head.
4. `train()` optimizes next-token prediction with cross-entropy loss.
5. `advise` detects a crop name in the query and prompts the model with that crop prefix. If no known crop is present, it uses the general-advice prefix.

## CLI commands

| Command | Purpose |
|---|---|
| `python -m agri_gpt train` | Train the tokenizer and GPT model |
| `python -m agri_gpt train --steps 1000` | Train for more update steps |
| `python -m agri_gpt advise tomato` | Generate tomato advice |
| `python -m agri_gpt advise "tomato yellow leaves"` | Generate advice with a crop/problem query |
| `python -m agri_gpt.evaluate` | Report held-out loss and perplexity |

## Testing

Run `python -m pytest -q`. Tests cover BPE round-tripping and serialization, model shapes and loss, context limits, training updates, checkpoint loading, CLI behavior, and evaluation output. GitHub Actions runs the test suite on pushes and pull requests.

## Evaluation and limitations

The evaluator uses the final 10% of the encoded corpus as a simple held-out segment and reports next-token loss and perplexity. Run it after training and copy the actual output into `RESULTS.md` if you want to report results. The corpus is small, so a low loss does not prove that advice is agronomically correct. The model can repeat text, produce incomplete wording, or fail on crops and conditions absent from training. Crop matching selects a crop prompt; it does not perform expert diagnosis. Add reviewed local material and evaluate with an agronomist before making stronger claims.

## Make it your own

Add locally relevant, verified snippets to `data/advice.txt` using the format `Crop: advice text`. Keep recommendations aligned with your region and crop stage. Retrain and rerun the evaluation after changing the corpus. Record where each recommendation came from if you extend this beyond a classroom demo.

## License

Add the license required by your course or repository before publishing.
## Model Training and Evaluation Results

The agricultural advice GPT model was trained for 500 steps using the project's agricultural text dataset.

### Training Progress

| Training Step | Training Loss |
|---:|---:|
| 100 | 8.2094 |
| 200 | 4.9275 |
| 300 | 4.1404 |
| 400 | 3.9113 |
| 500 | 3.6656 |

The training loss decreased consistently, indicating that the model learned patterns from the training data.

### Validation Results

| Metric | Result |
|---|---:|
| Validation Loss | 3.6974 |
| Perplexity | 40.344 |
| Validation Tokens | 656 |
| Training Steps | 500 |
| Automated Tests Passed | 20/20 |

### Sample Model Output

The model was tested with prompts related to three agricultural crops:

- Tomato
- Maize
- Wheat

The generated responses were not consistently coherent or reliable. This indicates that the current model has learned some text patterns but needs further training and improvement before it can provide useful agricultural recommendations.

### Model Checkpoint

The trained model checkpoint was saved at:

`checkpoints/agri_gpt.pt`

### Limitations and Future Improvements
## Project Architecture

This project implements a small GPT-style language model designed to generate agricultural text and advice.

### Main Components

1. **Agricultural Dataset:** Provides crop-related text examples used for model training.
2. **Custom Tokenizer:** A byte-level BPE tokenizer converts text into tokens that the model can process.
3. **GPT-style Transformer:** Learns language patterns using transformer-based architecture.
4. **Training Pipeline:** Trains the model using next-token prediction.
5. **Text Generation:** Generates responses based on agricultural prompts.
6. **Evaluation:** Measures validation loss and perplexity to assess model performance.

### Project Contributions

The project brings together the following implementation components:

- Agricultural text preparation and training examples.
- A custom byte-level BPE tokenizer.
- A GPT-style transformer training pipeline.
- Next-token prediction for text generation.
- Validation loss and perplexity evaluation.
- Automated tests for checking project functionality.

### Implementation and Attribution

This project is an educational implementation of a GPT-style language model for an agriculture-focused use case. It uses established concepts from transformer architecture and tokenization. Any external starter code, references, or libraries used are acknowledged in the relevant project files.

The agricultural dataset, training configuration, evaluation, and generated outputs should be considered in the context of this project's own implementation.

- The current generated text contains grammatical errors and incomplete recommendations.
- The training dataset is limited in size and diversity.
- Further training with a larger, high-quality agricultural dataset may improve generation quality.
- Generated agricultural advice should be verified against reliable agricultural sources before practical use.
# Agricultural Advice GPT — Evaluation Report

## 1. Training Summary

The GPT-style model was trained on agricultural text examples using next-token prediction.

- Total training steps: 500
- Model checkpoint: `checkpoints/agri_gpt.pt`

## 2. Training Loss

| Step | Loss |
|---:|---:|
| 100 | 8.2094 |
| 200 | 4.9275 |
| 300 | 4.1404 |
| 400 | 3.9113 |
| 500 | 3.6656 |

The training loss decreased throughout the training process.

## 3. Validation Metrics

| Metric | Value |
|---|---:|
| Validation Loss | 3.6974 |
| Perplexity | 40.344 |
| Validation Tokens | 656 |

Perplexity measures how uncertain the model is when predicting the next token. A lower value generally indicates better predictive performance.

## 4. Automated Testing

The project's automated test suite completed successfully:

- Tests passed: 20
- Tests failed: 0
- Test execution time: 11.20 seconds

## 5. Sample Generation Evaluation

The model was prompted with agricultural topics involving tomato, maize, and wheat.

The generated responses were not consistently coherent. Although the model produces text, its recommendations are not yet reliable enough for practical agricultural use.

## 6. Limitations

- Limited training data.
- Generated text may contain grammatical errors and incomplete information.
- Perplexity alone does not establish the factual correctness of agricultural advice.
- More diverse agricultural data and further model improvements are needed.

## 7. Reproducibility

The training and evaluation can be repeated using the setup and commands documented in the main README.

The saved checkpoint is located at `checkpoints/agri_gpt.pt`.
