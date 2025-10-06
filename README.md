## gpt.py – Small GPT-Style Character Transformer

### What it does
- Trains a compact GPT-like Transformer on `input.txt` at the character level.
- Learns long-range dependencies up to `block_size` using masked self-attention.
- Samples text autoregressively after training; optionally can dump long samples to `more.txt`.

### Architecture overview
- Token/position embeddings: `nn.Embedding(vocab_size, n_embd)` and `nn.Embedding(block_size, n_embd)`.
- Transformer blocks: `n_layer` stacks of pre-norm blocks with residuals.
  - `MultiHeadAttention`: parallel `Head`s, each with key/query/value projections and a causal mask.
  - `FeedFoward`: 2-layer MLP with expansion `4 * n_embd` and dropout.
- Final `LayerNorm` and linear head `lm_head` to logits over the vocabulary.
- Custom `_init_weights` for stable training.

### Important hyperparameters (defaults in file)
- `batch_size=64`, `block_size=256`, `n_embd=384`, `n_head=6`, `n_layer=6`, `dropout=0.2`, `learning_rate=3e-4`, `max_iters=5000`.

### Data & batching
- Builds a char vocabulary from `input.txt`; 90%/10% train/val split.
- `get_batch` draws random contiguous sequences of length `block_size` and their next-char targets.

### Training loop
- Optimizer: `AdamW`.
- Periodic evaluation on train/val via `estimate_loss()`.
- Prints parameter count at startup.

### Generation
- Crops the running context to the last `block_size` tokens and samples next chars via softmax.
- End of script prints a short sample; a longer sample write to `more.txt` is available but commented.

### How to run
```bash
cd ng-video-lecture
python gpt.py
```

Expected output: periodic train/val losses, parameter count, and a printed sample at the end.

### Notes
- Automatically uses GPU if available.
- For larger corpora or longer training, consider saving/loading checkpoints (not included in this minimal script).


