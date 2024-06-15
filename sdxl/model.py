# This file incorporates code from the following:
# Github Name                    | License | Link
# tinygrad/tinygrad              | MIT     | https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/LICENSE
# Stability-AI/generative-models | MIT     | https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/LICENSE-CODE
# mlfoundations/open_clip        | MIT     | https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/LICENSE

from tinygrad.tensor import Tensor # type: ignore
from tinygrad.nn import Linear, Conv2d, GroupNorm, LayerNorm, Embedding # type: ignore
from tinygrad.nn.state import safe_load # type: ignore
from tinygrad.helpers import fetch # type: ignore
from typing import Dict, List, Union, Callable, Optional, Any
from functools import lru_cache
import os, math, re, gzip

# configs:
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_base.yaml
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_refiner.yaml
configs: Dict = {
   "SDXL_Base": {
      "model": {"adm_in_channels": 2816, "in_channels": 4, "out_channels": 4, "model_channels": 320, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4], "d_head": 64, "transformer_depth": [1, 2, 10], "ctx_dim": 2048},
      "conditioner": {},
      "first_stage_model": {},
   },
   "SDXL_Refiner": {
      "model": {"adm_in_channels": 2560, "in_channels": 4, "out_channels": 4, "model_channels": 384, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4, 4], "d_head": 64, "transformer_depth": [4, 4, 4, 4], "ctx_dim": [1280, 1280, 1280, 1280]},
      "conditioner": {},
      "first_stage_model": {},
   }
}


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
         GEGLU(dim, dim*mult),
         lambda x: x,  # needed for weights loading code to work
         Linear(dim*mult, dim)
      ]

   def __call__(self, x:Tensor) -> Tensor:
      return x.sequential(self.net)


# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L200
class BasicTransformerBlock:
   def __init__(self, dim:int, ctx_dim:int, n_heads:int, d_head:int):
      self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
      self.ff = FeedForward(dim)
      self.attn2 = CrossAttention(dim, ctx_dim, n_heads, d_head)
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
      self.transformer_blocks = [BasicTransformerBlock(channels, ctx_dim[d], n_heads, d_head) for d in range(depth)]
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
               ResBlock(ch, time_embed_dim, model_channels*mult),
            ]
            ch = mult * model_channels
            if ds in attention_resolutions:
               n_heads = ch // d_head
               layers.append(SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[idx]))
            
            self.input_blocks.append(layers)
            input_block_channels.append(ch)
         
         if idx != len(channel_mult) - 1:
            self.input_blocks.append([
               Downsample(ch),
            ])
            ds *= 2
      
      n_heads = ch // d_head
      self.middle_block: List = [
         ResBlock(ch, time_embed_dim, ch),
         SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[-1]),
         ResBlock(ch, time_embed_dim, ch),
      ]

      self.output_blocks = []
      for idx, mult in list(enumerate(channel_mult))[::-1]:
         for i in range(self.num_res_blocks[idx] + 1):
            ich = input_block_channels.pop()
            layers = [
               ResBlock(ch + ich, time_embed_dim, model_channels*mult),
            ]
            
            if ds in attention_resolutions:
               n_heads = ch // d_head
               layers.append(SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[idx]))
            
            if idx > 0 and i == self.num_res_blocks[idx]:
               layers.append(Upsample(ch))
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
         if isinstance(bb, ResBlock): x = bb(x, emb)
         elif isinstance(bb, SpatialTransformer): x = bb(x, ctx)
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
   class ClosedClipTokenizer:
      def __init__(self):
         self.byte_encoder = Closed.bytes_to_unicode()
         merges: List[Any] = gzip.open(Closed.default_bpe()).read().decode("utf-8").split('\n')
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
            pairs = get_pairs(word)
         word = ' '.join(word)
         self.cache[token] = word
         return word

      def encode(self, text):
         bpe_tokens = []
         text = whitespace_clean(text.strip()).lower()
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
         self.mlp = Closed.ClosedClipMlp()
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


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L331
class FrozenClosedClipEmbedder:
   def __init__(self):
      self.tokenizer   = Closed.ClipTokenizer()
      self.transformer = Closed.ClipTextTransformer(layer_run_count=11+1)


class Open:
   """
   Namespace for OpenCLIP model components.
   """

   class MultiheadAttention:
      def __init__(self, dims:int, n_heads:int):
         self.dims     = dims
         self.n_heads  = n_heads
         self.d_head   = self.dims // self.n_heads
         self.k_proj   = Linear(self.dims, self.dims)
         self.v_proj   = Linear(self.dims, self.dims)
         self.q_proj   = Linear(self.dims, self.dims)
         self.out_proj = Linear(self.dims, self.dims)

      def __call__(self, x:Tensor, attn_mask:Optional[Tensor]=None) -> Tensor:
         B,L,D = x.shape
         q,k,v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
         q,k,v = [x.reshape(B, L, self.n_heads, self.d_head).transpose(1, 2) for x in (q,k,v)]
         attn_output = Tensor.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
         return self.out_proj(attn_output.transpose(1, 2).reshape(B, L, D))


   # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/transformer.py#L210
   class ResidualAttentionBlocks:
      def __init__(self, dims:int, n_heads:int, mlp_ratio:float):
         self.ln_1 = LayerNorm(dims)
         self.attn = Open.MultiheadAttention(dims, n_heads)
         self.ls_1 = lambda x: x

         self.ln_2 = LayerNorm(dims)
         d_mlp     = int(dims * mlp_ratio)
         self.mlp  = [
            Linear(dims, d_mlp),
            Tensor.gelu,
            Linear(d_mlp, dims),
         ]
         self.ls_2 = lambda x: x
      
      def __call__(self, x:Tensor, attn_mask:Optional[Tensor]=None) -> Tensor:
         x = x + x.sequential([self.ln_1, lambda z: self.attn(z, attn_mask), self.ls_1])
         x = x + x.sequential([self.ln_2, self.mlp, self.ls_2])
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


   # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/transformer.py#L661
   class ClipTextTransformer:
      def __init__(self, ctx_length:int=77, vocab_size:int=49408, dims:int=1024, n_heads:int=16, layers:int=24):
         self.token_embedding = Embedding(vocab_size, dims)
         self.positional_embedding = Tensor.empty(ctx_length, dims)
         self.transformer = Open.ClipTransformer(dims, layers, n_heads)


   # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/model.py#L220
   class Clip:
      def __init__(self):
         self.transformer = Open.ClipTextTransformer()

{
    "embed_dim": 1024,
   #  "text_cfg": {
   #      "context_length": 77,
   #      "vocab_size": 49408,
   #      "width": 1024,
   #      "heads": 16,
   #      "layers": 24
   #  }
}
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L396
class FrozenOpenClipEmbedder:
   pass


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L913
class ConcatTimestepEmbedderND:
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
   def __init__(self):
      self.embedders = []


class FirstStageModel:
   pass


class SDXL:
   def __init__(self, config:Dict):
      self.model = UNetModel(**config["model"])
      self.conditioner = Conditioner(**config["conditioner"])
      self.first_stage_model = FirstStageModel(**config["first_stage_model"])

if __name__ == "__main__":
   weight_path = os.path.join(os.path.dirname(__file__), "..", "weights", "sd_xl_base_1.0.safetensors")
   d = safe_load(weight_path)
