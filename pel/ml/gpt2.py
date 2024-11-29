"""
Notes on "Let's reproduce GPT-2 (124M)"
https://www.youtube.com/watch?v=l8pRSuU81PU

Building GPT-2 but referencing the GPT-3 paper in addition for finer details.
  - GPT-2 Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  - GPT-3 Paper: https://arxiv.org/pdf/2005.14165

Karpathy's Tiny Shakespeare dataset is available at:
https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt

Saved local copy of the dataset:
~/Datasets/tinyshakespeare/input.txt
"""

from dataclasses import dataclass
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# -----------------------------------------------------------------------------


@dataclass
class GPT2Config:
    block_size: int = 1024  # max context length
    vocab_size: int = 50257  # GPT-2 vocab size (50k BPE merges, 256 bytes tokens, 1 end token)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768  # embedding dimensionality


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads" and hs is "head size", and C = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Attention (materializes the large TxT matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)
        # Flash Attention:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)  # (B, T, C)
        return y


class MLP(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()  # Original paper uses tanh approximation
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    A GPT-2 block. Differs from prior work by moving layernorm to the input of each sub-block
    and keeping a "clean" residual stream.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # Reduce op
        x = x + self.mlp(self.ln_2(x))  # Map op
        return x


class GPT(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init parameters
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            # More generally, one would prefer to use Xavier initialization that scales the stddev of the weights by 1/sqrt(d)
            # But this follows the OpenAI GPT-2 source code.
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: None | torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the GPT model
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)  # (B, T, n_embd)
        x = self.transformer.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay: float, learning_rate: float, device: str):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Do not decay the bias and LayerNorm parameters
        decay_params = [p for p in param_dict.values() if p.ndim >= 2]
        nodecay_params = [p for p in param_dict.values() if p.ndim < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed: {num_decay_params}, num non-decayed: {num_nodecay_params}")
        # Create AdamW optimizer and use the fused kernel
        use_fused = "cuda" in device
        print(f"using fused optimizer: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type: str):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head, n_embd are determined from model_type
        config_args = {
            "gpt2":         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 token vocab for GPT-2
        config_args["block_size"] = 1024  # always 1024 context length for GPT-2
        # Create a from-scratch initialized model
        config = GPT2Config(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]  # discard the mask buffers
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(".attn.bias")]  # discard the mask buffers
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(".attn.masked_bias")]  # discard the mask buffers
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        # The openai checkpoints use a "Conv1D" module but we want to use a Linear module, so we need to transpose
        assert len(sd_keys) == len(sd_hf_keys), f"mismatch in number of keys: {len(sd_keys)} != {len(sd_hf_keys)}"

        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                # special handling for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                sd[k].copy_(sd_hf[k].t())
            else:
                # everything else can be a simple copy
                assert sd_hf[k].shape == sd[k].shape
                sd[k].copy_(sd_hf[k])

        return model


class DataLoaderLite:
    def __init__(self, B: int, T: int):
        self.B = B
        self.T = T

        with open("/home/jackson/Datasets/tinyshakespeare/input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch has {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0
    
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # Advance the position
        self.current_position += B * T
        # if loading the next batch would exceed the number of tokens, reset the position
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y


# -----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T) == 0, f"total_batch_size must be divisible by (B * T): {total_batch_size} % {B * T} != 0"
grad_accum_steps = total_batch_size // (B * T)

# model = GPT.from_pretrained("gpt2")
model = GPT(GPT2Config(vocab_size=50304))  # Add 'fake' tokens to reach nearest multiple of 128
model.eval()
model.to(device)
model = torch.compile(model)

torch.set_float32_matmul_precision("high")

train_loader = DataLoaderLite(B=B, T=T)

# 3090Ti Benchmark:

# BF16 w/ torch.compile and Flash Attention and 'nice' vocab size:
# B = 10, T = 1024  --> 66.80 tokens/second

# BF16 w/ torch.compile and Flash Attention:
# B = 10, T = 1024  --> 65.25 tokens/second

# BF16 w/ torch.compile:
# B = 10, T = 1024  --> 53.35 tokens/second
# B = 4, T = 1024   --> 43.35 tokens/second
# B = 1, T = 1024   --> 27.70 tokens/second

# BF16:
# B = 10, T = 1024  --> 31.50 tokens/second
# B = 4, T = 1024   --> 28.20 tokens/second
# B = 1, T = 1024   --> 19.95 tokens/second

# TF32:
# B = 10, T = 1024  --> 22.65 tokens/second
# B = 4, T = 1024   --> 20.75 tokens/second
# B = 1, T = 1024   --> 16.30 tokens/second

# FP32:
# B = 10, T = 1024  --> 17.95 tokens/second
# B = 4, T = 1024   --> 16.70 tokens/second
# B = 1, T = 1024   --> 13.20 tokens/second

max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 50
def get_lr(it: int):
    # Linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # Constant after full schedule
    if it > max_steps:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
    t0 = time.time()
    loss_accum = 0.0
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach().cpu()
            loss.backward()
    
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()  # wait for all kernels to finish
    t1 = time.time()
    dt = (t1 - t0) * 1000  # milliseconds
    tokens_per_second = (train_loader.B * train_loader.T * grad_accum_steps) / dt
    print(f"step {step:4d} | loss: {loss_accum.item():7.4f} | lr: {lr:10.3e} | norm: {norm:6.2f} | {dt:6.2f} ms/batch | {tokens_per_second:7.2f} tokens/second")

import sys; sys.exit()

# generate! right now x is (B, T) where B is the number of sequences to generate
# set the seed to 42
num_return_sequences = 5
max_length = 30

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]  # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # k=50 is the HF default
        ix = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
        xcol = torch.gather(topk_indices, dim=-1, index=ix)  # (B, 1)
        x = torch.cat((x, xcol), dim=1)  # (B, T+1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
