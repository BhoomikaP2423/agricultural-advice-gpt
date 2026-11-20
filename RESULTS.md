# Evaluation results

No benchmark numbers are claimed before training. Run the following after training to generate actual held-out metrics:

```bash
python -m agri_gpt train --steps 500
python -m agri_gpt.evaluate
```

The evaluator reserves the final 10% of the encoded corpus for next-token loss and perplexity. This is a small demonstration corpus, so these metrics are only a basic training check; they do not establish field-level agricultural accuracy or generalization to unseen regions and crops.
