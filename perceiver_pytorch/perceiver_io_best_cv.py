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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)

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


class GRUGating(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, **kwargs):
        b, dim = x.shape[0], self.dim
        y = self.fn(x, **kwargs)

        gated_output = self.gru(
            rearrange(y, '... d -> (...) d'),
            rearrange(x, '... d -> (...) d')
        )

        gated_output = rearrange(gated_output, '(b n) d -> b n d', b = b)
        return gated_output

# main class

class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth = 20,
        dim = 128,
        input_queries_dim = 128,
        output_queries_dim = 128,
        latent_queries_dim = 128,
        num_output_queries = 32,
        num_latents = 64,
        logits_dim = 60,
        input_cross_heads = 4,
        cross_heads = 4,
        latent_heads = 8,
        input_cross_dim_head = 64,
        cross_dim_head = 64,
        latent_queries_dim_head = 64,
        weight_tie_layers = True
    ):
        super().__init__()
        
        self.max_steps = depth
        self.lambda_prob = nn.Sigmoid()
        self.logits_dim = logits_dim
                
        # if learn_latents:
        self.latents = nn.Parameter(torch.randn(num_latents, latent_queries_dim))

        # more cross heads here
        get_query_cross_attn = lambda: PreNorm(input_queries_dim, Attention(input_queries_dim, dim, heads = input_cross_heads, dim_head=input_cross_dim_head))
        get_query_cross_ff = lambda: PreNorm(input_queries_dim, FeedForward(input_queries_dim))

        get_cross_attn = lambda: PreNorm(latent_queries_dim, Attention(latent_queries_dim, input_queries_dim, heads = cross_heads, dim_head = cross_dim_head))
        get_cross_ff = lambda: PreNorm(latent_queries_dim, FeedForward(latent_queries_dim))
        # get_latent_attn = lambda: PreNorm(latent_queries_dim, Attention(latent_queries_dim, heads = latent_heads, dim_head = latent_queries_dim_head))
        # get_latent_ff = lambda: PreNorm(latent_queries_dim, FeedForward(latent_queries_dim))
        get_latent_attn = lambda: PreNorm(latent_queries_dim, GRUGating(latent_queries_dim, Residual(PreNorm(latent_queries_dim, Attention(latent_queries_dim, heads = latent_heads, dim_head = latent_queries_dim_head)))))
        get_decoder_ca = lambda: PreNorm(output_queries_dim, Attention(output_queries_dim, latent_queries_dim, heads = cross_heads, dim_head = cross_dim_head))
        get_decoder_ff = lambda: PreNorm(output_queries_dim, FeedForward(output_queries_dim))
        get_to_logits = lambda: nn.Linear(num_output_queries * output_queries_dim, logits_dim + 1) if exists(logits_dim) else lambda: nn.Identity()

        # get_latent_attn, get_latent_ff, get_decoder_ca, get_to_logits = map(cache_fn, (get_latent_attn, get_latent_ff, get_decoder_ca, get_to_logits))
        get_latent_attn, get_decoder_ca, get_decoder_ff, get_to_logits = map(cache_fn, (get_latent_attn, get_decoder_ca, get_decoder_ff, get_to_logits))

        self.encoder_cross_attn = nn.ModuleList([
            get_query_cross_attn(),
            get_query_cross_ff(),
            get_cross_attn(),
            get_cross_ff()
        ])
        
        self.layers = nn.ModuleList([])
        for i in range(depth):
            cache_args = {'_cache': weight_tie_layers}

            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_decoder_ca(**cache_args),
                get_decoder_ff(**cache_args),
                get_to_logits(**cache_args)
            ]))

    def forward(
        self,
        context,
        mask = None,
        input_queries = None,
        output_queries = None
    ):
        b, *_, device = *context.shape, context.device

        p = []
        y = []
        un_halted_prob = context.new_ones((b, 1))
        halted = context.new_zeros((b, 1))
        p_m = context.new_zeros((b, 1))
        y_m = context.new_zeros((b, self.logits_dim))

        latents = repeat(self.latents, 'n d -> b n d', b = b)
        query_cross_attn, query_cross_ff, cross_attn, cross_ff = self.encoder_cross_attn
        x = query_cross_attn(input_queries, context = context, mask = mask) + input_queries
        x = query_cross_ff(x) + x
        x = cross_attn(latents, context = context, mask = mask) + latents
        x = cross_ff(x) + x
        
        for n in range(self.max_steps): 
            self_attns, decoder_cross_attn, decoder_ff, to_logits = self.layers[n]  
            x = self_attns(x) + x
            # if not exists(queries):
            #     latents = x
            # cross attend from decoder queries to latents
            latents = decoder_cross_attn(output_queries, context = x) # torch.squeeze()
            latents = decoder_ff(latents) + latents
            latents = rearrange(latents, 'b n d -> b (n d)')
            logits = to_logits(latents)
            
            if n + 1 == self.max_steps:
                lambda_n = context.new_ones(b, 1)
            else:
                lambda_n = self.lambda_prob(logits[:, 0])[:, None]
    
            y_n = logits[:, 1:]
            p_n = un_halted_prob * lambda_n
            un_halted_prob = un_halted_prob * (1 - lambda_n)
            halt = torch.bernoulli(lambda_n) * (1 - halted)
            
            p.append(p_n)
            y.append(y_n)
        
            p_m = p_m * (1 - halt) + p_n * halt
            y_m = y_m * (1 - halt) + y_n * halt
        
            halted = halted + halt
            
            if not self.training and halted.sum() == b:
                break
            
        return torch.squeeze(torch.stack(p)), torch.stack(y), p_m, y_m


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_func: nn.Module):
        super().__init__()
        self.loss_func = loss_func
        
    def forward(self, p: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        total_loss = p.new_tensor(0.)
        for n in range(p.shape[0]):
            loss = (p[n] * self.loss_func(y_hat[n], y)).sum()
            total_loss = total_loss + loss
        return total_loss
    
        
class RegularizationLoss(nn.Module):        
    def __init__(self, lambda_p: float, max_steps: int):    
        super().__init__()
        p_g = torch.zeros((max_steps,))
        not_halted = 1.
        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)    
        self.p_g = nn.Parameter(p_g, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction='sum')
        
    def forward(self, p: torch.Tensor):
        p = torch.squeeze(p).transpose(0, 1)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)
        return self.kl_div((p).log(), p_g)
    
    
class PonderLoss(nn.Module):
    def __init__(self, loss_func: nn.Module, lambda_p: float, max_steps: int, beta: float):
        super().__init__()
        self.loss_rec = ReconstructionLoss(loss_func)
        self.loss_reg = RegularizationLoss(lambda_p, max_steps)
        self.beta = beta
    
    def forward(self, p: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        return self.loss_rec(p, y_hat, y) + self.beta * self.loss_reg(p)
    
    
# # Perceiver LM example
# class PerceiverLM(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         num_tokens,
#         max_seq_len,
#         **kwargs
#     ):
#         super().__init__()
#         self.token_emb = nn.Embedding(num_tokens, dim)
#         self.pos_emb = nn.Embedding(max_seq_len, dim)

#         self.perceiver_io = PerceiverIO(
#             dim = dim,
#             queries_dim = dim,
#             logits_dim = num_tokens,
#             **kwargs
#         )

#     def forward(
#         self,
#         x,
#         mask = None
#     ):
#         n, device = x.shape[1], x.device
#         x = self.token_emb(x)

#         pos_emb = self.pos_emb(torch.arange(n, device = device))
#         pos_emb = rearrange(pos_emb, 'n d -> () n d')
#         x = x + pos_emb

#         logits = self.perceiver_io(x, mask = mask, queries = x)
#         return logits


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
        num_output_queries,
        output_queries_dim,
        **kwargs
    ):
        super().__init__()
        
        self.answer_query = torch.nn.Parameter(torch.randn(1, num_output_queries, output_queries_dim))  
        
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.context_pe = PositionalEncoding(d_model=dim, max_len=context_max_seq_len, dropout=0)        
        self.question_pe = PositionalEncoding(d_model=dim, max_len=question_max_seq_len, dropout=0)        

        self.perceiver_io = PerceiverIO(dim = dim, output_queries_dim=output_queries_dim, num_output_queries=num_output_queries, **kwargs)

    def forward(
        self,
        contexts,
        questions
    ):
        contexts = self.token_emb(contexts)
        contexts = self.context_pe(contexts)  
        
        questions = self.token_emb(questions)
        questions = self.question_pe(questions)
        
        answer_queries = self.answer_query.expand(contexts.size(0), self.answer_query.size(1), self.answer_query.size(2))
        logits = self.perceiver_io(contexts, input_queries=questions, output_queries=answer_queries)
        return logits
    
# class PerceiverIObAbInq(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim,
#         num_tokens,
#         context_max_seq_len,
#         **kwargs
#     ):
#         super().__init__()
        
#         self.answer_query = torch.nn.Parameter(torch.randn(1, 1, num_tokens))  # (1, 1, num_tokens)
        
#         self.token_emb = nn.Embedding(num_tokens, dim)
#         self.context_pe = PositionalEncoding(d_model=dim, max_len=context_max_seq_len, dropout=0)        
#         # self.question_pe = PositionalEncoding(d_model=dim, max_len=question_max_seq_len, dropout=0)        

#         self.perceiver_io = PerceiverIO(dim = dim, **kwargs)

#     def forward(
#         self,
#         contexts
#     ):
#         contexts = self.token_emb(contexts)
#         contexts = self.context_pe(contexts)  
        
#         # questions = self.token_emb(questions)
#         # questions = self.question_pe(questions)
        
#         answer_queries = self.answer_query.expand(contexts.size(0), self.answer_query.size(1), self.answer_query.size(2)) # (?, 1, num_tokens)
        
#         logits = self.perceiver_io(contexts, queries=answer_queries)
#         return logits
    
