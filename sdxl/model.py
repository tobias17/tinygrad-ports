# This file incorporates code from the following:
# Github Name                    | License | Link
# tinygrad/tinygrad              | MIT     | https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/LICENSE
# Stability-AI/generative-models | MIT     | https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/LICENSE-CODE
# mlfoundations/open_clip        | MIT     | https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/LICENSE

from tinygrad.tensor import Tensor, dtypes # type: ignore
from tinygrad.nn import Linear, Conv2d, GroupNorm, LayerNorm, Embedding # type: ignore
from tinygrad.nn.state import safe_load, load_state_dict # type: ignore
from tinygrad.helpers import fetch # type: ignore
from typing import Dict, List, Union, Callable, Optional, Any, Set
from functools import lru_cache
import os, math, re, gzip
from PIL import Image # type: ignore
from abc import ABC
import numpy as np

# configs:
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_base.yaml
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_refiner.yaml
configs: Dict = {
   "SDXL_Base": {
      "model": {"adm_in_channels": 2816, "in_channels": 4, "out_channels": 4, "model_channels": 320, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4], "d_head": 64, "transformer_depth": [1, 2, 10], "ctx_dim": 2048},
      "conditioner": {},
      "first_stage_model": {"ch": 128, "in_ch": 3, "out_ch": 3, "z_ch": 4, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "resolution": 256},
   },
   "SDXL_Refiner": {
      "model": {"adm_in_channels": 2560, "in_channels": 4, "out_channels": 4, "model_channels": 384, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4, 4], "d_head": 64, "transformer_depth": [4, 4, 4, 4], "ctx_dim": [1280, 1280, 1280, 1280]},
      "conditioner": {},
      "first_stage_model": {"ch": 128, "in_ch": 3, "out_ch": 3, "z_ch": 4, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "resolution": 256},
   }
}


class UNet:
   """
   Namespace for UNet model components.
   """

   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L136
   class ResBlock:
      def __init__(self, channels:int, emb_channels:int, out_channels:int):
         self.in_layers = [
            GroupNorm(32, channels),
            Tensor.silu,
            Conv2d(channels, out_channels, 3, padding=1),
         ]
         self.emb_layers = [
            Tensor.silu,
            Linear(emb_channels, out_channels),
         ]
         self.out_layers = [
            GroupNorm(32, out_channels),
            Tensor.silu,
            lambda x: x,  # needed for weights loading code to work
            Conv2d(out_channels, out_channels, 3, padding=1),
         ]
         self.skip_connection = Conv2d(channels, out_channels, 1) if channels != out_channels else lambda x: x

      def __call__(self, x:Tensor, emb:Tensor) -> Tensor:
         h = x.sequential(self.in_layers)
         emb_out = emb.sequential(self.emb_layers)
         h = h + emb_out.reshape(*emb_out.shape, 1, 1)
         h = h.sequential(self.out_layers)
         return self.skip_connection(x) + h


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L163
   class CrossAttention:
      def __init__(self, query_dim, context_dim, n_heads, d_head):
         self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
         self.to_k = Linear(context_dim, n_heads*d_head, bias=False)
         self.to_v = Linear(context_dim, n_heads*d_head, bias=False)
         self.num_heads = n_heads
         self.head_size = d_head
         self.to_out = [Linear(n_heads*d_head, query_dim)]

      def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
         ctx = x if ctx is None else ctx
         q,k,v = self.to_q(x), self.to_k(ctx), self.to_v(ctx)
         q,k,v = [y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(1,2) for y in (q,k,v)]
         attention = Tensor.scaled_dot_product_attention(q, k, v).transpose(1,2)
         h_ = attention.reshape(x.shape[0], -1, self.num_heads * self.head_size)
         return h_.sequential(self.to_out)


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L180
   class GEGLU:
      def __init__(self, dim_in:int, dim_out:int):
         self.proj = Linear(dim_in, dim_out * 2)
         self.dim_out = dim_out

      def __call__(self, x:Tensor) -> Tensor:
         x, gate = self.proj(x).chunk(2, dim=-1)
         return x * gate.gelu()


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L189
   class FeedForward:
      def __init__(self, dim:int, mult:int=4):
         self.net = [
            UNet.GEGLU(dim, dim*mult),
            lambda x: x,  # needed for weights loading code to work
            Linear(dim*mult, dim)
         ]

      def __call__(self, x:Tensor) -> Tensor:
         return x.sequential(self.net)


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L200
   class BasicTransformerBlock:
      def __init__(self, dim:int, ctx_dim:int, n_heads:int, d_head:int):
         self.attn1 = UNet.CrossAttention(dim, dim, n_heads, d_head)
         self.ff = UNet.FeedForward(dim)
         self.attn2 = UNet.CrossAttention(dim, ctx_dim, n_heads, d_head)
         self.norm1 = LayerNorm(dim)
         self.norm2 = LayerNorm(dim)
         self.norm3 = LayerNorm(dim)

      def __call__(self, x, context=None):
         x = self.attn1(self.norm1(x)) + x
         x = self.attn2(self.norm2(x), context=context) + x
         x = self.ff(self.norm3(x)) + x
         return x


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L215
   # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/attention.py#L619
   class SpatialTransformer:
      def __init__(self, channels:int, n_heads:int, d_head:int, ctx_dim:Union[int,List[int]], depth:int=1):
         if isinstance(ctx_dim, int):
            ctx_dim = [ctx_dim]*depth
         else:
            assert isinstance(ctx_dim, list) and depth == len(ctx_dim)
         self.norm = GroupNorm(32, channels)
         assert channels == n_heads * d_head
         self.proj_in = Linear(channels, n_heads * d_head)
         self.transformer_blocks = [UNet.BasicTransformerBlock(channels, ctx_dim[d], n_heads, d_head) for d in range(depth)]
         self.proj_out = Linear(n_heads * d_head, channels)

      def __call__(self, x:Tensor, context:Optional[Tensor]=None) -> Tensor:
         b, c, h, w = x.shape
         x_in = x
         x = self.norm(x)
         x = x.reshape(b, c, h*w).permute(0,2,1)
         x = self.proj_in(x)
         for block in self.transformer_blocks:
            x = block(x, context=context)
         x = self.proj_out(x)
         x = x.permute(0,2,1).reshape(b, c, h, w)
         return x + x_in


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L235
   class Downsample:
      def __init__(self, channels:int):
         self.op = Conv2d(channels, channels, 3, stride=2, padding=1)

      def __call__(self, x:Tensor) -> Tensor:
         return self.op(x)


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L242
   class Upsample:
      def __init__(self, channels:int):
         self.conv = Conv2d(channels, channels, 3, padding=1)

      def __call__(self, x:Tensor) -> Tensor:
         bs,c,py,px = x.shape
         x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
         return self.conv(x)


# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L251
def timestep_embedding(timesteps, dim, max_period=10000):
   half = dim // 2
   freqs = (-math.log(max_period) * Tensor.arange(half) / half).exp()
   args = timesteps * freqs
   return Tensor.cat(args.cos(), args.sin()).reshape(1, -1)


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/openaimodel.py#L472
# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L257
class UNetModel:
   def __init__(self, adm_in_channels:int, in_channels:int, out_channels:int, model_channels:int, attention_resolutions:List[int], num_res_blocks:int, channel_mult:List[int], d_head:int, transformer_depth:List[int], ctx_dim:Union[int,List[int]]):
      self.in_channels = in_channels
      self.model_channels = model_channels
      self.out_channels = out_channels
      self.num_res_blocks = [num_res_blocks] * len(channel_mult)

      self.attention_resolutions = attention_resolutions
      self.dropout = 0.0
      self.channel_mult = channel_mult
      self.conv_resample = True
      self.num_classes = None
      self.use_checkpoint = False
      self.d_head = d_head

      time_embed_dim = model_channels * 4
      self.time_embed = [
         Linear(model_channels, time_embed_dim),
         Tensor.silu,
         Linear(time_embed_dim, time_embed_dim),
      ]

      self.label_emb = [
         Linear(adm_in_channels, time_embed_dim),
         Tensor.silu,
         Linear(time_embed_dim, time_embed_dim),
      ]

      self.input_blocks = [
         [Conv2d(in_channels, model_channels, 3, padding=1)]
      ]
      input_block_channels = [model_channels]
      ch = model_channels
      ds = 1
      for idx, mult in enumerate(channel_mult):
         for _ in range(self.num_res_blocks[idx]):
            layers: List[Any] = [
               UNet.ResBlock(ch, time_embed_dim, model_channels*mult),
            ]
            ch = mult * model_channels
            if ds in attention_resolutions:
               n_heads = ch // d_head
               layers.append(UNet.SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[idx]))
            
            self.input_blocks.append(layers)
            input_block_channels.append(ch)
         
         if idx != len(channel_mult) - 1:
            self.input_blocks.append([
               UNet.Downsample(ch),
            ])
            ds *= 2
      
      n_heads = ch // d_head
      self.middle_block: List = [
         UNet.ResBlock(ch, time_embed_dim, ch),
         UNet.SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[-1]),
         UNet.ResBlock(ch, time_embed_dim, ch),
      ]

      self.output_blocks = []
      for idx, mult in list(enumerate(channel_mult))[::-1]:
         for i in range(self.num_res_blocks[idx] + 1):
            ich = input_block_channels.pop()
            layers = [
               UNet.ResBlock(ch + ich, time_embed_dim, model_channels*mult),
            ]
            
            if ds in attention_resolutions:
               n_heads = ch // d_head
               layers.append(UNet.SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[idx]))
            
            if idx > 0 and i == self.num_res_blocks[idx]:
               layers.append(UNet.Upsample(ch))
            self.output_blocks.append(layers)

      self.out = [
         GroupNorm(32, ch),
         Tensor.silu,
         Conv2d(self.out_channels, self.out_channels, 3, padding=1),
      ]

   def __call__(self, x:Tensor, tms:Tensor, ctx:Tensor, y:Tensor) -> Tensor:
      t_emb = timestep_embedding(tms, self.model_channels)
      emb = t_emb.sequential(self.time_embed)

      def run(x:Tensor, bb) -> Tensor:
         if isinstance(bb, UNet.ResBlock): x = bb(x, emb)
         elif isinstance(bb, UNet.SpatialTransformer): x = bb(x, ctx)
         else: x = bb(x)
         return x

      saved_inputs = []
      for b in self.input_blocks:
         for bb in b:
            x = run(x, bb)
         saved_inputs.append(x)
      for bb in self.middle_block:
         x = run(x, bb)
      for b in self.output_blocks:
         x = x.cat(saved_inputs.pop(), dim=1)
         for bb in b:
            x = run(x, bb)
      return x.sequential(self.out)


class Embedder(ABC):
   input_key: str


class Closed:
   """
   Namespace for OpenAI CLIP model components.
   """

   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L409
   @staticmethod
   def get_pairs(word):
      """
      Return set of symbol pairs in a word.
      Word is represented as tuple of symbols (symbols being variable-length strings).
      """
      return set(zip(word, word[1:]))
   @staticmethod
   def whitespace_clean(text):
      text = re.sub(r'\s+', ' ', text)
      text = text.strip()
      return text
   @staticmethod
   def bytes_to_unicode():
      """
      Returns list of utf-8 byte and a corresponding list of unicode strings.
      The reversible bpe codes work on unicode strings.
      This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
      When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
      This is a significant percentage of your normal, say, 32K bpe vocab.
      To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
      And avoids mapping to whitespace/control characters the bpe code barfs on.
      """
      bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
      cs = bs[:]
      n = 0
      for b in range(2**8):
         if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
      cs = [chr(n) for n in cs]
      return dict(zip(bs, cs))
   # Clip tokenizer, taken from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py (MIT license)
   @lru_cache()
   @staticmethod
   def default_bpe(): return fetch("https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz", "bpe_simple_vocab_16e6.txt.gz")
   class ClipTokenizer:
      def __init__(self):
         self.byte_encoder = Closed.bytes_to_unicode()
         merges = gzip.open(Closed.default_bpe()).read().decode("utf-8").split('\n')
         merges = merges[1:49152-256-2+1]
         merges = [tuple(merge.split()) for merge in merges]
         vocab = list(Closed.bytes_to_unicode().values())
         vocab = vocab + [v+'</w>' for v in vocab]
         for merge in merges:
            vocab.append(''.join(merge))
         vocab.extend(['<|startoftext|>', '<|endoftext|>'])
         self.encoder = dict(zip(vocab, range(len(vocab))))
         self.bpe_ranks = dict(zip(merges, range(len(merges))))
         self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
         self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\s]+""", re.IGNORECASE)

      def bpe(self, token):
         if token in self.cache:
            return self.cache[token]
         word = tuple(token[:-1]) + ( token[-1] + '</w>',)
         pairs = Closed.get_pairs(word)

         if not pairs:
            return token+'</w>'

         while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
               break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
               try:
                  j = word.index(first, i)
                  new_word.extend(word[i:j])
                  i = j
               except Exception:
                  new_word.extend(word[i:])
                  break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
               new_word.append(first+second)
               i += 2
            else:
               new_word.append(word[i])
               i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
               break
            pairs = Closed.get_pairs(word)
         word = ' '.join(word)
         self.cache[token] = word
         return word

      def encode(self, text):
         bpe_tokens = []
         text = Closed.whitespace_clean(text.strip()).lower()
         for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
         # Truncation, keeping two slots for start and end tokens.
         if len(bpe_tokens) > 75:
            bpe_tokens = bpe_tokens[:75]
         return [49406] + bpe_tokens + [49407] * (77 - len(bpe_tokens) - 1)


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L329
   class ClipMlp:
      def __init__(self):
         self.fc1 = Linear(768, 3072)
         self.fc2 = Linear(3072, 768)

      def __call__(self, hidden_states):
         hidden_states = self.fc1(hidden_states)
         hidden_states = hidden_states.quick_gelu()
         hidden_states = self.fc2(hidden_states)
         return hidden_states


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L340
   class ClipAttention:
      def __init__(self):
         self.embed_dim = 768
         self.num_heads = 12
         self.head_dim = self.embed_dim // self.num_heads
         self.k_proj = Linear(self.embed_dim, self.embed_dim)
         self.v_proj = Linear(self.embed_dim, self.embed_dim)
         self.q_proj = Linear(self.embed_dim, self.embed_dim)
         self.out_proj = Linear(self.embed_dim, self.embed_dim)

      def __call__(self, hidden_states, causal_attention_mask):
         bsz, tgt_len, embed_dim = hidden_states.shape
         q,k,v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
         q,k,v = [x.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2) for x in (q,k,v)]
         attn_output = Tensor.scaled_dot_product_attention(q, k, v, attn_mask=causal_attention_mask)
         return self.out_proj(attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim))


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L357
   class ClipEncoderLayer:
      def __init__(self):
         self.self_attn = Closed.ClipAttention()
         self.layer_norm1 = LayerNorm(768)
         self.mlp = Closed.ClipMlp()
         self.layer_norm2 = LayerNorm(768)

      def __call__(self, hidden_states:Tensor, causal_attention_mask:Tensor) -> Tensor:
         residual = hidden_states
         hidden_states = self.layer_norm1(hidden_states)
         hidden_states = self.self_attn(hidden_states, causal_attention_mask)
         hidden_states = residual + hidden_states

         residual = hidden_states
         hidden_states = self.layer_norm2(hidden_states)
         hidden_states = self.mlp(hidden_states)
         hidden_states = residual + hidden_states

         return hidden_states


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L377
   class ClipEncoder:
      def __init__(self, layer_run_count:Optional[int]=None):
         self.layers = [Closed.ClipEncoderLayer() for i in range(12)]
         self.layer_run_count = layer_run_count

      def __call__(self, hidden_states:Tensor, causal_attention_mask:Tensor) -> Tensor:
         for l in self.layers[:self.layer_run_count]:
            hidden_states = l(hidden_states, causal_attention_mask)
         return hidden_states


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L386
   class ClipTextEmbeddings:
      def __init__(self):
         self.token_embedding    = Embedding(49408, 768)
         self.position_embedding = Embedding(77, 768)

      def __call__(self, input_ids:Tensor, position_ids:Tensor) -> Tensor:
         return self.token_embedding(input_ids) + self.position_embedding(position_ids)


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L394
   class ClipTextTransformer:
      def __init__(self, layer_run_count:Optional[int]=None):
         self.embeddings       = Closed.ClipTextEmbeddings()
         self.encoder          = Closed.ClipEncoder(layer_run_count)
         self.final_layer_norm = LayerNorm(768)
         self.layer_run_count  = layer_run_count

      def __call__(self, input_ids):
         x = self.embeddings(input_ids, Tensor.arange(input_ids.shape[1]).reshape(1, -1))
         x = self.encoder(x, Tensor.full((1, 1, 77, 77), float("-inf")).triu(1))
         return self.final_layer_norm(x) if self.layer_run_count is None else x
   
   class ClipTextModel:
      def __init__(self):
         self.text_model = Closed.ClipTextTransformer(layer_run_count=11+1)


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L331
class FrozenClosedClipEmbedder(Embedder):
   def __init__(self):
      self.tokenizer   = Closed.ClipTokenizer()
      self.transformer = Closed.ClipTextModel()
      self.input_key = "txt"


class Open:
   """
   Namespace for OpenCLIP model components.
   """

   class MultiheadAttention:
      def __init__(self, dims:int, n_heads:int):
         self.dims     = dims
         self.n_heads  = n_heads
         self.d_head   = self.dims // self.n_heads

         self.in_proj_bias   = Tensor.empty(3*dims)
         self.in_proj_weight = Tensor.empty(3*dims, dims)
         self.out_proj = Linear(dims, dims)

      def __call__(self, x:Tensor, attn_mask:Optional[Tensor]=None) -> Tensor:
         B,L,D = x.shape
         q_b, k_b, v_b = self.in_proj_bias  .chunk(3, dim=0)
         q_w, k_w, v_w = self.in_proj_weight.chunk(3, dim=0)
         q,k,v = (x @ q_w) + q_b, (x @ k_w) + k_b, (x @ v_w) + v_b
         q,k,v = [x.reshape(B, L, self.n_heads, self.d_head).transpose(1, 2) for x in (q,k,v)]
         attn_output = Tensor.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
         return self.out_proj(attn_output.transpose(1, 2).reshape(B, L, D))


   class Mlp:
      def __init__(self, dims, hidden_dims):
         self.c_fc   = Linear(dims, hidden_dims)
         self.c_proj = Linear(hidden_dims, dims)
      
      def __call__(self, x:Tensor) -> Tensor:
         return x.sequential([self.c_fc, Tensor.gelu, self.c_proj])


   # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/transformer.py#L210
   class ResidualAttentionBlocks:
      def __init__(self, dims:int, n_heads:int, mlp_ratio:float):
         self.ln_1 = LayerNorm(dims)
         self.attn = Open.MultiheadAttention(dims, n_heads)

         self.ln_2 = LayerNorm(dims)
         self.mlp  = Open.Mlp(dims, int(dims * mlp_ratio))
      
      def __call__(self, x:Tensor, attn_mask:Optional[Tensor]=None) -> Tensor:
         x = x + x.sequential([self.ln_1, lambda z: self.attn(z, attn_mask)])
         x = x + x.sequential([self.ln_2, self.mlp])
         return x


   # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/transformer.py#L317
   class ClipTransformer:
      def __init__(self, dims:int, layers:int, n_heads:int, mlp_ratio:float=4.0):
         self.resblocks = [
            Open.ResidualAttentionBlocks(dims, n_heads, mlp_ratio) for _ in range(layers)
         ]
      
      def __call__(self, x:Tensor, attn_mask:Optional[Tensor]=None) -> Tensor:
         x = x.transpose(0, 1).contiguous()
         for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
         x = x.transpose(0, 1)
         return x


   # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/model.py#L220
   # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/transformer.py#L661
   class ClipTextTransformer:
      def __init__(self, dims:int=1024, vocab_size:int=49408, n_heads:int=16, ctx_length:int=77, layers:int=24):
         self.token_embedding = Embedding(vocab_size, dims)
         self.positional_embedding = Tensor.empty(ctx_length, dims)
         self.transformer = Open.ClipTransformer(dims, layers, n_heads)
         self.ln_final = LayerNorm(dims)
         self.text_projection = Tensor.empty(dims, 512)
      
      @property
      def attn_mask(self) -> Tensor:
         if not hasattr(self, "_attn_mask"):
            self._attn_mask = Tensor.full((1, 1, 77, 77), float("-inf")).triu(1)
         return self._attn_mask

      def __call__(self, text:Tensor) -> Tensor:
         seq_len = text.shape[1]

         x = self.token_embedding(text)
         x = x + self.positional_embedding[:seq_len]
         x = self.transformer(x, attn_mask=self.attn_mask)
         x = self.ln_final(x)

         pooled = x[Tensor.arange(x.shape[0]), text.argmax(dim=-1)]
         pooled = pooled @ self.text_projection
         return pooled


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L396
class FrozenOpenClipEmbedder(Embedder):
   def __init__(self, dims:int=1024):
      self.model = Open.ClipTextTransformer()
      self.input_key = "txt"


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L913
class ConcatTimestepEmbedderND(Embedder):
   def __init__(self, outdim:int, input_key:str):
      self.outdim = outdim
      self.input_key = input_key

   def __call__(self, x:Tensor):
      assert len(x.shape) == 2
      b, _ = x.shape
      x = x.flatten()
      emb = timestep_embedding(x, self.outdim)
      emb = x.reshape((b,-1))
      return emb


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L71
class Conditioner:
   OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
   KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

   def __init__(self):
      self.embedders: List[Embedder] = [FrozenClosedClipEmbedder(), FrozenOpenClipEmbedder()]
   
   def get_keys(self) -> Set[str]:
      return set(e.input_key for e in self.embedders)
   
   def __call__(self, batch:Dict) -> Dict[str,Tensor]:
      output: Dict[str,Tensor] = {}

      for embedder in self.embedders:
         with Tensor.no_grad():
            emb_out = embedder(batch[embedder.input_key])

         if isinstance(emb_out, Tensor):
            emb_out = [emb_out]
         else:
            assert isinstance(emb_out, (list, tuple))

         for emb in emb_out:
            out_key = self.OUTPUT_DIM2KEYS[len(emb.shape)]
            if out_key in output:
               output[out_key] = output[out_key].cat(emb, dim=self.KEY2CATDIM[out_key])
            else:
               output[out_key] = emb

      return output


class FirstStage:
   """
   Namespace for First Stage Model components
   """

   # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L94
   class ResnetBlock:
      def __init__(self, in_dim, out_dim):
         pass
      
      def __call__(self, x:Tensor) -> Tensor:
         return x # FIXME


   # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L74
   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L102
   class Downsample:
      def __init__(self, dims:int):
         self.conv = Conv2d(dims, dims, kernel_size=3, stride=2, padding=(0,1,0,1))

      def __call__(self, x:Tensor) -> Tensor:
         return self.conv(x)


   # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L58
   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L83
   class Upsample:
      def __init__(self, dims:int):
         self.conv = Conv2d(dims, dims, kernel_size=3, stride=1, padding=1)

      def __call__(self, x:Tensor) -> Tensor:
         B,C,Y,X = x.shape
         x = x.reshape(B, C, Y, 1, X, 1).expand(B, C, Y, 2, X, 2).reshape(B, C, Y*2, X*2)
         return self.conv(x)


   # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L204
   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L17
   class AttnBlock:
      def __init__(self, in_channels):
         self.norm = GroupNorm(32, in_channels)
         self.q = Conv2d(in_channels, in_channels, 1)
         self.k = Conv2d(in_channels, in_channels, 1)
         self.v = Conv2d(in_channels, in_channels, 1)
         self.proj_out = Conv2d(in_channels, in_channels, 1)

      # copied from AttnBlock in ldm repo
      def __call__(self, x:Tensor) -> Tensor:
         h_ = self.norm(x)
         q,k,v = self.q(h_), self.k(h_), self.v(h_)

         # compute attention
         b,c,h,w = q.shape
         q,k,v = [x.reshape(b,c,h*w).transpose(1,2) for x in (q,k,v)]
         h_ = Tensor.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,c,h,w)
         return x + self.proj_out(h_)


   class MidEntry:
      def __init__(self, block_in:int):
         self.block_1 = FirstStage.ResnetBlock(block_in, block_in),
         self.attn_1  = FirstStage.AttnBlock  (block_in),
         self.block_2 = FirstStage.ResnetBlock(block_in, block_in),


   # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L487
   class Encoder:
      def __init__(self, ch:int, in_ch:int, out_ch:int, z_ch:int, ch_mult:List[int], num_res_blocks:int, resolution:int):
         self.conv_in = Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
         in_ch_mult = (1,) + tuple(ch_mult)

         class BlockEntry:
            def __init__(self, block:List[FirstStage.ResnetBlock], downsample:Callable[[Tensor],Tensor]):
               self.block = block
               self.downsample = downsample
         self.down: List[BlockEntry] = []
         for i_level in range(len(ch_mult)):
            block = []
            block_in  = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult   [i_level]
            for i_block in range(num_res_blocks):
               block.append(FirstStage.ResnetBlock(block_in, block_out))
               block_in = block_out
            
            downsample = lambda x: x if i_level == len(ch_mult)-1 else FirstStage.Downsample(block_in)
            self.down.append(BlockEntry(block, downsample))
         
         self.mid = FirstStage.MidEntry(block_in)

         self.norm_out = GroupNorm(32, in_ch)
         self.conv_out = Conv2d(block_in, 2*z_ch, kernel_size=3, stride=1, padding=1)
      
      def __call__(self, x:Tensor) -> Tensor:
         h = self.conv_in(x)
         for down in self.down:
            for block in down.block:
               h = block(h)
            h = down.downsample(h)
         
         h = h.sequential([self.mid.block_1, self.mid.attn_1, self.mid.block_2])
         h = h.sequential([self.norm_out,    Tensor.swish,    self.conv_out   ])
         return h


   # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L604
   class Decoder:
      def __init__(self, ch:int, in_ch:int, out_ch:int, z_ch:int, ch_mult:List[int], num_res_blocks:int, resolution:int):
         block_in = ch * ch_mult[-1]
         curr_res = resolution // 2 ** (len(ch_mult) - 1)
         self.z_shape = (1, z_ch, curr_res, curr_res)
         
         self.conv_in = Conv2d(z_ch, block_in, kernel_size=3, stride=1, padding=1)

         self.mid = FirstStage.MidEntry(block_in)

         class BlockEntry:
            def __init__(self, block:List[FirstStage.ResnetBlock], upsample:Callable[[Tensor],Tensor]):
               self.block = block
               self.upsample = upsample
         self.up: List[BlockEntry] = []
         for i_level in reversed(range(len(ch_mult))):
            block = []
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
               block.append(FirstStage.ResnetBlock(block_in, block_out))
               block_in = block_out
            
            upsample = lambda x: x if i_level == 0 else FirstStage.Upsample(block_in)
            self.up.insert(0, BlockEntry(block, upsample))
         
         self.norm_out = GroupNorm(32, in_ch)
         self.conv_out = Conv2d(block_in, 2*z_ch, kernel_size=3, stride=1, padding=1)
      
      def __call__(self, z:Tensor) -> Tensor:
         self.last_z_shape = z.shape
         
         h = z.sequential([self.conv_in, self.mid.block_1, self.mid.attn_1, self.mid.block_2])
         
         for up in self.up:
            for block in up.block:
               h = block(h)
            h = up.upsample(h)

         h = h.sequential([self.norm_out, Tensor.swish, self.conv_out])
         return h


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L102
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L437
class FirstStageModel:
   def __init__(self, embed_dim:int=4, **kwargs):
      self.encoder = FirstStage.Encoder(**kwargs)
      self.decoder = FirstStage.Decoder(**kwargs)
      self.quant_conv = Conv2d(2*kwargs["z_ch"], 2*embed_dim, 1)
      self.post_quant_conv = Conv2d(embed_dim, kwargs["z_ch"], 1)
   
   def __call__(self, x:Tensor) -> Tensor:
      return x.sequential([self.encoder, self.quant_conv, lambda l: l[:,0:4], self.post_quant_conv, self.decoder])


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/diffusion.py#L19
class SDXL:
   scale_factor: float = 0.13025

   def __init__(self, config:Dict):
      self.conditioner = Conditioner(**config["conditioner"])
      self.first_stage_model = FirstStageModel(**config["first_stage_model"])
      self.model = UNetModel(**config["model"])
   
   def denoiser(model, x:Tensor, sigma, c) -> Tensor:
      pass

   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L543
   def decode(self, x:Tensor) -> Tensor:
      x = self.first_stage_model(1.0/self.scale_factor * x)

      # make image correct size and scale
      x = (x + 1.0) / 2.0
      x = x.reshape(3,512,512).permute(1,2,0).clip(0,1)*255
      return x.cast(dtypes.uint8)


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/sampling.py#L287
def dpmpp2m_sampler(denoiser, x, c, uc, steps, cfg_scale):
   pass


if __name__ == "__main__":
   weight_path = os.path.join(os.path.dirname(__file__), "..", "weights", "sd_xl_base_1.0.safetensors")
   state_dict = safe_load(weight_path)

   model = SDXL(configs["SDXL_Base"])
   load_state_dict(model, state_dict, strict=False)

   # sampling params
   # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/api.py#L52
   pos_prompt = "a horse sized cat eating a bagel"
   neg_prompt = ""
   img_width  = 1024
   img_height = 1024
   steps = 50
   cfg_scale = 6.0
   eta = 1.0
   aesthetic_score = 5.0

   N = 1
   C = 4
   F = 8

   # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/helpers.py
   batch_c : Dict = {
      "txt": np.array([pos_prompt]).repeat(N,1),
      "original_size_as_tuple": Tensor((img_height,img_width)).repeat(N,1),
      "crop_coords_top_left": Tensor((0,0)).repeat(N,1),
      "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
   }
   batch_uc: Dict = {
      "txt": np.array([neg_prompt]).repeat(N,1),
      "original_size_as_tuple": Tensor((img_height,img_width)).repeat(N,1),
      "crop_coords_top_left": Tensor((0,0)).repeat(N,1),
      "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
   }
   c, uc = model.conditioner(batch_c), model.conditioner(batch_uc)

   shape = (N, C, img_height // F, img_width // F)
   randn = Tensor.randn(shape)

   def denoiser(x, sigma, c) -> Tensor:
      return model.denoiser(model.model, x, sigma, c)

   z = dpmpp2m_sampler(denoiser, randn, c, uc, steps, cfg_scale)
   x = model.decode(z)

   print(x.shape)
   
   im = Image.fromarray(x.numpy().astype(np.uint8, copy=False))
   im.show()




