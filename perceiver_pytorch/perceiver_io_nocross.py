from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import Variable

from einops import rearrange, repeat
import math

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        self_per_cross_attn = 1,
        learn_latents=True
    ):
        super().__init__()
        if learn_latents:
            self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_decoder_ca = lambda: PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        get_to_logits = lambda: nn.Linear(queries_dim, logits_dim + 1) if exists(logits_dim) else lambda: nn.Identity()

        get_latent_attn, get_latent_ff, get_decoder_ca, get_to_logits = map(cache_fn, (get_latent_attn, get_latent_ff, get_decoder_ca, get_to_logits))

        self.encoder_cross_attn = nn.ModuleList([
            get_cross_attn(),
            get_cross_ff()
        ])
        
        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ])

            decoder_self_atnns  = nn.ModuleList([
                get_decoder_ca(**cache_args),
                get_to_logits(**cache_args)
            ])
            
            self.layers.append(nn.ModuleList([
                self_attns,
                decoder_self_atnns
            ]))

    def forward(
        self,
        data,
        mask = None,
        latents = None,
        queries = None
    ):
        b, *_, device = *data.shape, data.device

        if latents is not None:
            x = latents
        else:
            x = repeat(self.latents, 'n d -> b n d', b = b)

        cross_attn, cross_ff = self.encoder_cross_attn
        x = cross_attn(x, context = data, mask = mask) + x
        x = cross_ff(x) + x
        # layers

        for n in range(len(self.layers)):
            self_attns, decoder_self_atnns = self.layers[n]
            self_attn, self_ff = self_attns
            x = self_attn(x) + x
            x = self_ff(x) + x       
            
        if not exists(queries):
                latents = x
        # cross attend from decoder queries to latents
        decoder_cross_attn, to_logits = decoder_self_atnns
        latents = torch.squeeze(decoder_cross_attn(queries, context = x))
        logits = to_logits(latents)

        return logits

# Perceiver LM example

class PerceiverLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver_io = PerceiverIO(
            dim = dim,
            queries_dim = dim,
            logits_dim = num_tokens,
            **kwargs
        )

    def forward(
        self,
        x,
        mask = None
    ):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        logits = self.perceiver_io(x, mask = mask, queries = x)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = torch.transpose(x, 0, 1)
        x = x + self.pe[:x.size(0)]
        x = torch.transpose(x, 0, 1)
        return self.dropout(x)


class PerceiverIObAbI(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        context_max_seq_len,
        question_max_seq_len,
        **kwargs
    ):
        super().__init__()
        
        self.answer_query = torch.nn.Parameter(torch.randn(1, 1, num_tokens))  # (1, 1, num_tokens)
        
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.context_pe = PositionalEncoding(d_model=dim, max_len=context_max_seq_len, dropout=0)        
        self.question_pe = PositionalEncoding(d_model=dim, max_len=question_max_seq_len, dropout=0)        

        self.perceiver_io = PerceiverIO(dim = dim, **kwargs)

    def forward(
        self,
        contexts,
        questions
    ):
        contexts = self.token_emb(contexts)
        contexts = self.context_pe(contexts)  
        
        questions = self.token_emb(questions)
        questions = self.question_pe(questions)
        
        answer_queries = self.answer_query.expand(contexts.size(0), self.answer_query.size(1), self.answer_query.size(2)) # (?, 1, num_tokens)
        
        logits = self.perceiver_io(contexts, latents=questions, queries=answer_queries)
        return logits
    
class PerceiverIObAbInq(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        context_max_seq_len,
        **kwargs
    ):
        super().__init__()
        
        self.answer_query = torch.nn.Parameter(torch.randn(1, 1, num_tokens))  # (1, 1, num_tokens)
        
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.context_pe = PositionalEncoding(d_model=dim, max_len=context_max_seq_len, dropout=0)        
        # self.question_pe = PositionalEncoding(d_model=dim, max_len=question_max_seq_len, dropout=0)        

        self.perceiver_io = PerceiverIO(dim = dim, **kwargs)

    def forward(
        self,
        contexts
    ):
        contexts = self.token_emb(contexts)
        contexts = self.context_pe(contexts)  
        
        # questions = self.token_emb(questions)
        # questions = self.question_pe(questions)
        
        answer_queries = self.answer_query.expand(contexts.size(0), self.answer_query.size(1), self.answer_query.size(2)) # (?, 1, num_tokens)
        
        logits = self.perceiver_io(contexts, queries=answer_queries)
        return logits
    
    
class PonderPerceiver(nn.Module):
    def __init__(self, max_steps, n_hidden, **kwargs):
        self.max_steps = max_steps
        self.n_hidden = n_hidden
        super().__init__()
        
        
        
        self.perceiver_io = PerceiverIObAbInq(**kwargs)
        
    def forward(self, x):
        return self.perceiver_io(x)