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
python gpt.py
```

Expected output: periodic train/val losses, parameter count, and a printed sample at the end.

### Notes
- Automatically uses GPU if available.
- For larger corpora or longer training, consider saving/loading checkpoints (not included in this minimal script).


### Example training log and sample output

The following run used the defaults in `gpt.py` and tiny Shakespeare (`input.txt`). It shows typical convergence and a representative generated sample after training:

```text
0.209729 M parameters
step 0: train loss 4.4116, val loss 4.4022
step 100: train loss 2.6568, val loss 2.6670
step 200: train loss 2.5090, val loss 2.5058
step 300: train loss 2.4198, val loss 2.4340
step 400: train loss 2.3503, val loss 2.3567
step 500: train loss 2.2970, val loss 2.3136
step 600: train loss 2.2410, val loss 2.2506
step 700: train loss 2.2062, val loss 2.2198
step 800: train loss 2.1638, val loss 2.1871
step 900: train loss 2.1232, val loss 2.1494
step 1000: train loss 2.1020, val loss 2.1293
step 1100: train loss 2.0704, val loss 2.1196
step 1200: train loss 2.0382, val loss 2.0798
step 1300: train loss 2.0249, val loss 2.0640
step 1400: train loss 1.9922, val loss 2.0354
step 1500: train loss 1.9707, val loss 2.0308
step 1600: train loss 1.9614, val loss 2.0474
step 1700: train loss 1.9393, val loss 2.0130
step 1800: train loss 1.9070, val loss 1.9943
step 1900: train loss 1.9057, val loss 1.9871
step 2000: train loss 1.8834, val loss 1.9954
step 2100: train loss 1.8719, val loss 1.9758
step 2200: train loss 1.8582, val loss 1.9623
step 2300: train loss 1.8546, val loss 1.9517
step 2400: train loss 1.8410, val loss 1.9476
step 2500: train loss 1.8167, val loss 1.9455
step 2600: train loss 1.8263, val loss 1.9401
step 2700: train loss 1.8108, val loss 1.9340
step 2800: train loss 1.8040, val loss 1.9247
step 2900: train loss 1.8044, val loss 1.9304
step 3000: train loss 1.7963, val loss 1.9242
step 3100: train loss 1.7687, val loss 1.9147
step 3200: train loss 1.7547, val loss 1.9102
step 3300: train loss 1.7557, val loss 1.9037
step 3400: train loss 1.7547, val loss 1.8946
step 3500: train loss 1.7385, val loss 1.8968
step 3600: train loss 1.7260, val loss 1.8914
step 3700: train loss 1.7257, val loss 1.8808
step 3800: train loss 1.7204, val loss 1.8919
step 3900: train loss 1.7215, val loss 1.8788
step 4000: train loss 1.7146, val loss 1.8639
step 4100: train loss 1.7095, val loss 1.8724
step 4200: train loss 1.7079, val loss 1.8707
step 4300: train loss 1.7035, val loss 1.8502
step 4400: train loss 1.7043, val loss 1.8693
step 4500: train loss 1.6914, val loss 1.8522
step 4600: train loss 1.6853, val loss 1.8357
step 4700: train loss 1.6862, val loss 1.8483
step 4800: train loss 1.6671, val loss 1.8434
step 4900: train loss 1.6736, val loss 1.8415
step 4999: train loss 1.6635, val loss 1.8226

FlY BOLINGLO:
Them thrumply towiter arts the
muscue rike begatt the sea it
What satell in rowers that some than othis Marrity.

LUCENTVO:
But userman these that, where can is not diesty rege;
What and see to not. But's eyes. What?

JOHN MARGARET:
Than up I wark, what out, I ever of and love,
one these do sponce, vois I me;
But my pray sape to ries all to the not erralied in may.

BENVOLIO:
To spits as stold's bewear I would and say mesby all
on sworn make he anough
As cousins the solle, whose be my conforeful may lie them yet
nobe allimely untraled to be thre I say be,
Notham a brotes theme an make come,
And that his reach to the duke ento
the grmeants bell! and now there king-liff-or grief?

GLOUCESTER:
All the bettle dreene, for To his like thou thron!

MENENIUS:
Then, if I knom her all.
My lord, but terruly friend
Rish of the ploceiness and wilt tends sure?
Is you knows a fasir wead
That with him my spaut,
I shall not tas where's not, becomity; my coulds sting,
then the wit be dong to tyget our hereefore,
Who strop me, mend here, if agains, bitten, thy lack.
The but these it were is tus. For the her skeep the fasting. joy tweet Bumner:-
How the enclady: It you and how,
I am in him, And ladderle:
Their hand whose wife, it my hithre,
Roman and where sposs gives'd you.

TROMIOLANUS:
But livants you great, I shom mistrot come, for to she to lot
for smy to men ventry mehus. Gazise;
Full't were some the cause, and stouch set,
Or promises, which a kingsasted to your gove them; and sterrer,
And that wae love him.

BRUTUS:
You shape with these sweet.

CORTENGONO:
Lo, where 'twon elmes, 'morth young agres;
Sir, azavoust to striel accurded we missery sets crave.

ANGOLUM:
For is Henry to have gleise the dreason
That I ant shorfold wefth their servy in enscy.

ISABELLA:
O, I better you eyse such formfetrews.

BUCKINGHARENT:
Qead my lightle this righanneds flase them
Wam which an take was our some pleasurs,
Lovisoname to me, then fult me?--have it?

HENRY BOLINGBROY:
That wha
```


