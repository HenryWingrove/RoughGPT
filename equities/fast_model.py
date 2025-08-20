"""
This code is based on nanoGPT by Karpathy: https://github.com/karpathy/nanoGPT
... and on the gpt-fast repo from pytorch: https://github.com/pytorch-labs/gpt-fast/
.... and on the llama2.c repo by Karpathy: https://github.com/karpathy/llama2.c/

Full definition of a modern GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) summary of the Meta AI Llama2 architecture, a modern GPT variant:
https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7
"""

import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

KVCACHE = True

@dataclass
class ModelArgs:
    # inspired by default hyperparameters for the Llama 7B model
    dim: int = 768 # embedding dimension
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    vocab_size: int = 12515
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 10368 # block size
    dropout: float = 0.0
    sig_dim: int = 0
    use_signatures: bool = False


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) replaces LayerNorm in the original
    implementation. RMSNorm retains the re-scaling invariance property (without the
    overhead of re-centering properties) while simplifying the computation.
    """
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


"""
Rotary Position Embedding (RoPE) are used in place of traditional absolute positional
encoding. RoPE provides several benefits over traditional positional encoding, including
flexibility in sequence length, decaying inter-token dependencies, and enhanced self-attention.

For more details: https://blog.eleuther.ai/rotary-embeddings/
"""
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_rotary_emb_single(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape x to match the complex representation
    x_r, x_i = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, x_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, x_r)

    # apply rotation using real numbers
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # flatten last two dimensions
    x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(3)

    return x_out.type_as(x)

"""
Grouped-Query Attention (GQA) maintains a balance between performance and computational
efficiency by grouping queries and attending to a subset of keys and values. This is a 
key component of the Llama2 architecture, and is used in place of traditional self-attention.
"""
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class KVCache:
    """
    Key-Value (KV) caching is used to store the key and value tensors for each token
    in the sequence. This allows for efficient computation of self-attention, as the
    key and value tensors are only computed once and then reused for each token.

    This technique is used to accelerate the inference process in the GPT model. The
    trade-off is that it requires additional GPU memory to store the key and value states.
    """
    def __init__(self, shape, max_seq_length,device=None,dtype=None):
        self.key: torch.Tensor = torch.zeros(shape, device=device, dtype=dtype)
        self.value: torch.Tensor = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq_length = max_seq_length # hard code "true" cache limit for now?
        self.encoded_tok_len = 24 # also functions as roll_len when using bpe encoding
        self.sink_tokens = 1 # 0 # 24

    def update(
        self, keys: torch.Tensor, values: torch.Tensor, start_pos: torch.Tensor, roll: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz,T,nh,C = keys.shape
        if roll:
            sink_side_key = self.key[:, 0:self.sink_tokens, :, :]
            recent_side_key = (self.key[:, self.sink_tokens:, :, :]).roll(-self.encoded_tok_len, dims=1)
            self.key = torch.cat((sink_side_key, recent_side_key), dim=1)
            sink_side_value = self.value[:, 0:self.sink_tokens, :, :]
            recent_side_value = (self.value[:, self.sink_tokens:, :, :]).roll(-self.encoded_tok_len, dims=1)
            self.value = torch.cat((sink_side_value, recent_side_value), dim=1)
        self.key[:bsz, start_pos : start_pos + T] = keys
        self.value[:bsz, start_pos : start_pos + T] = values
        keys = self.key[:bsz, : start_pos + T]
        values = self.value[:bsz, : start_pos + T]
        return keys, values
    

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        # query, key, value projections
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # output projection
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        cache:KVCache=None,
        input_pos:torch.Tensor=None,
        roll:bool=False,
        freqs_cache:Tuple[torch.Tensor,torch.Tensor]=None
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # # RoPE relative positional embeddings
        # xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        if cache is None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        else:
            assert input_pos is not None
            xk,xv = cache.update(xk,xv,input_pos,roll)
            # apply relative positional embeddings (RoPE)
            xq = apply_rotary_emb_single(xq, freqs_cos, freqs_sin)
            freqs_cos_cache, freqs_sin_cache = freqs_cache
            xk = apply_rotary_emb_single(xk, freqs_cos_cache, freqs_sin_cache)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # self-attention
        if self.flash and cache is None:
            # efficient attention using Flash Attention CUDA kernels
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    """
    This FeedForward class now consists of three linear transformations and, like Llama2,
    makes use of the Sigmoid Linear Unit (SiLU; also known as swish) activation function.
    
    SiLU consistently performs slightly better then GELU across a range of experiments, and
    in some implementations is more efficient: https://arxiv.org/pdf/1710.05941.pdf
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, cache, input_pos, roll, freqs_cache):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin, cache=cache, input_pos=input_pos, roll=roll, freqs_cache=freqs_cache)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        if params.use_signatures:
            self.sig_proj = nn.Linear(params.sig_dim, params.dim)
        else:
            self.sig_proj = None
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # key-value cache for fast inference
        self.kv_cache = [None for _ in range(self.params.n_layers)]

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model"""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: bool = False,
        max_seq_length: int = None,
        input_pos: torch.Tensor = None,
        roll: bool = False,
        sigs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        tok_emb = self.tok_embeddings(tokens)
        if self.params.use_signatures:
            if sigs is None:
                raise ValueError("sigs must be provided when use_signatures is True")
            h = tok_emb + self.sig_proj(sigs)
        else:
            h = tok_emb
        h = self.dropout(h)
        if kv_cache:
            freqs_cos = self.freqs_cos[input_pos:input_pos+seqlen]
            freqs_sin = self.freqs_sin[input_pos:input_pos+seqlen]
            freqs_cache = (self.freqs_cos[:input_pos+seqlen], self.freqs_sin[:input_pos+seqlen])
        else:
            freqs_cos = self.freqs_cos[:seqlen]
            freqs_sin = self.freqs_sin[:seqlen]
            freqs_cache = None
    
        if kv_cache:
            assert max_seq_length is not None
            assert input_pos is not None
            if self.kv_cache[0] is None:
                shape = (_bsz,max_seq_length,self.params.n_kv_heads,self.params.dim//self.params.n_heads)
                self.kv_cache = [KVCache(shape,max_seq_length,device=h.device,dtype=h.dtype) for _ in range(self.params.n_layers)]
        elif self.kv_cache[-1] is not None:
            self.kv_cache = [None for _ in range(self.params.n_layers)]

        for layer,cache in zip(self.layers,self.kv_cache):
            h = layer(h, freqs_cos, freqs_sin, cache, input_pos, roll, freqs_cache)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of RTX 4090 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of RTX 4090 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 165e12 # RTX 4090 is cited to be 165 TFLOPS (330 TFLOPS with sparsity feature) of bloat16 running on tensor cores
        mfu = flops_achieved / flops_promised
        return mfu
    
    def relevant_mask(self, tok_pos, logits, vocab_encoding, device):
        # determine field and special tokens to include
        include_nan = False
        if tok_pos == 0:
            field = 'ticker'
        elif tok_pos == 1:
            field = 'type'
        elif tok_pos == 2:
            field = 'side'
        elif tok_pos in [3, 16]:
            field = 'sign'
            include_nan = True
        elif tok_pos in [4, 17]:
            field = 'price'
            include_nan = True
        elif tok_pos in [5, 6, 18]:
            field = 'size'
            include_nan = True
        elif tok_pos in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
            field = 'time'
        elif tok_pos in [19, 20, 21, 22, 23]:
            field = 'time'
            include_nan = True

        # select relevant tokens, excluding special tokens
        if not include_nan:
            relevant_indices = vocab_encoding[field][1][3:]
        else:
            relevant_indices = vocab_encoding[field][1][2:]
        # prepare tensors for scattering
        relevant_destination = torch.zeros(1, logits.shape[1], device=device)
        relevant_indices = torch.tensor(relevant_indices, dtype=torch.int64, device=device).unsqueeze(0)
        relevant_src = torch.ones(1, relevant_indices.shape[1], device=device)
        # place the values in relevant_src at the indices in relevant_indices in relevant_destination
        relevant_mask = relevant_destination.scatter(1, index=relevant_indices, src=relevant_src).bool()
        # mask the indices in logits that are not relevant for the current field
        logits.masked_fill_(~relevant_mask, float("-inf"))
        return logits

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p = 0.9,
                 start=False, roll=False, new_block_size=10344, vocab_encoding=None,
                 use_relevant_mask=True, use_bpe=False, eom_token_val=0, roll_len=24):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            idx: (b,t) LongTensor of input tokens
            max_new_tokens: int, the number of tokens to generate
            temperature >0: scale logits before applying softmax
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            start: bool, if True, the input is the start of a new sequence, otherwise it is a continuation
            roll: bool, if True, the input is the start of a rolling sequence and the KV cache is rolled
            new_block_size: int, the new block size for the KV cache when rolling
            vocab_encoding: dict, the encoding of the vocabulary
            use_relevant_mask: bool, if True, mask out irrelevant tokens for the current field
            use_bpe: bool, if True, use BPE encoding
            eom_token_val: int, the end-of-message token value
            roll_len: int, the length of the roll when using BPE encoding
        """
        B,T = idx.shape
        input_pos = idx.shape[1] - 1
        for tok_pos in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            if KVCACHE and not start:
                x = idx_cond[:,-1].reshape(-1,1)
            else:
                x = idx_cond
                input_pos = 0
            if use_bpe and roll:
                for kv_cache in self.kv_cache:
                    kv_cache.encoded_tok_len = roll_len + 1
            logits = self.forward(x, None, kv_cache=KVCACHE, max_seq_length=new_block_size, input_pos=input_pos, roll=roll)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0: # greedy sampling:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally mask indices that are not relevant for the current token field
                if use_relevant_mask and not use_bpe:
                    logits = self.relevant_mask(tok_pos, logits, vocab_encoding, logits.device)
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                if top_p > 0.0:
                    # First sort and calculate cumulative sum of probabilities.
                    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
                    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
                    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                    # scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits.masked_fill_(indices_to_remove, float("-inf"))
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            input_pos = idx.shape[1]
            start = False
            roll = False
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next[0].item() == eom_token_val:
                return idx, tok_pos

        if use_bpe:
            return idx, tok_pos
        else:
            return idx