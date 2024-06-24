# This file incorporates code from the following:
# Github Name                    | License | Link
# tinygrad/tinygrad              | MIT     | https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/LICENSE
# Stability-AI/generative-models | MIT     | https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/LICENSE-CODE
# mlfoundations/open_clip        | MIT     | https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/LICENSE

from tinygrad.tensor import Tensor, dtypes # type: ignore
from tinygrad.nn import Linear, Conv2d, GroupNorm, LayerNorm, Embedding # type: ignore
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict # type: ignore
from tinygrad.helpers import fetch # type: ignore
from tinygrad import TinyJit
from typing import Dict, List, Union, Callable, Optional, Any, Set, Tuple
from abc import ABC, abstractmethod
from functools import lru_cache
import os, math, re, gzip
from tqdm import trange
from PIL import Image # type: ignore
import numpy as np

# configs:
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_base.yaml
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_refiner.yaml
configs: Dict = {
   "SDXL_Base": {
      "model": {"adm_in_channels": 2816, "in_channels": 4, "out_channels": 4, "model_channels": 320, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4], "d_head": 64, "transformer_depth": [1, 2, 10], "ctx_dim": 2048},
      "conditioner": {"concat_embedders": ["original_size_as_tuple", "crop_coords_top_left", "target_size_as_tuple"]},
      "first_stage_model": {"ch": 128, "in_ch": 3, "out_ch": 3, "z_ch": 4, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "resolution": 256},
      "denoiser": {"num_idx": 1000},
   },
   "SDXL_Refiner": {
      "model": {"adm_in_channels": 2560, "in_channels": 4, "out_channels": 4, "model_channels": 384, "attention_resolutions": [4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 4, 4], "d_head": 64, "transformer_depth": [4, 4, 4, 4], "ctx_dim": [1280, 1280, 1280, 1280]},
      "conditioner": {"concat_embedders": ["original_size_as_tuple", "crop_coords_top_left", "aesthetic_score"]},
      "first_stage_model": {"ch": 128, "in_ch": 3, "out_ch": 3, "z_ch": 4, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "resolution": 256},
      "denoiser": {"num_idx": 1000},
   }
}


def tensor_identity(x:Tensor) -> Tensor:
   return x


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
         self.skip_connection = Conv2d(channels, out_channels, 1) if channels != out_channels else tensor_identity

      def __call__(self, x:Tensor, emb:Tensor) -> Tensor:
         h = x.sequential(self.in_layers)
         emb_out = emb.sequential(self.emb_layers)
         h = h + emb_out.reshape(*emb_out.shape, 1, 1)
         h = h.sequential(self.out_layers)
         return self.skip_connection(x) + h


   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L163
   class CrossAttention:
      def __init__(self, query_dim:int, ctx_dim:int, n_heads:int, d_head:int):
         self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
         self.to_k = Linear(ctx_dim,   n_heads*d_head, bias=False)
         self.to_v = Linear(ctx_dim,   n_heads*d_head, bias=False)
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
         self.ff    = UNet.FeedForward(dim)
         self.attn2 = UNet.CrossAttention(dim, ctx_dim, n_heads, d_head)
         self.norm1 = LayerNorm(dim)
         self.norm2 = LayerNorm(dim)
         self.norm3 = LayerNorm(dim)

      def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
         x = self.attn1(self.norm1(x)) + x
         x = self.attn2(self.norm2(x), ctx=ctx) + x
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

      def __call__(self, x:Tensor, ctx:Optional[Tensor]=None) -> Tensor:
         b, c, h, w = x.shape
         x_in = x
         x = self.norm(x)
         x = x.reshape(b, c, h*w).permute(0,2,1) # b c h w -> b c (h w) -> b (h w) c
         x = self.proj_in(x)
         for block in self.transformer_blocks:
            x = block(x, ctx=ctx)
         x = self.proj_out(x)
         x = x.permute(0,2,1).reshape(b, c, h, w) # b (h w) c -> b c (h w) -> b c h w
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
         z = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
         return self.conv(z)


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/util.py#L207
# https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L251
def timestep_embedding(timesteps, dim, max_period=10000):
   half = dim // 2
   freqs = (-math.log(max_period) * Tensor.arange(half) / half).exp()
   args = timesteps[:, None].cast(dtypes.float32) * freqs[None]
   return Tensor.cat(args.cos(), args.sin(), dim=-1).cast(dtypes.float16)


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
      self.num_classes = "sequential"
      self.use_checkpoint = False
      self.d_head = d_head

      time_embed_dim = model_channels * 4
      self.time_embed = [
         Linear(model_channels, time_embed_dim),
         Tensor.silu,
         Linear(time_embed_dim, time_embed_dim),
      ]

      self.label_emb = [
         [
            Linear(adm_in_channels, time_embed_dim),
            Tensor.silu,
            Linear(time_embed_dim, time_embed_dim),
         ]
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
            input_block_channels.append(ch)
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
            ch = model_channels * mult
            
            if ds in attention_resolutions:
               n_heads = ch // d_head
               layers.append(UNet.SpatialTransformer(ch, n_heads, d_head, ctx_dim, depth=transformer_depth[idx]))
            
            if idx > 0 and i == self.num_res_blocks[idx]:
               layers.append(UNet.Upsample(ch))
               ds //= 2
            self.output_blocks.append(layers)

      self.out = [
         GroupNorm(32, ch),
         Tensor.silu,
         Conv2d(model_channels, out_channels, 3, padding=1),
      ]

   def __call__(self, x:Tensor, tms:Tensor, c:Dict) -> Tensor:
      ctx = c.get("crossattn", None)
      y   = c.get("vector", None)
      cat = c.get("concat", None)
      if cat is not None:
         x = x.cat(cat, dim=1)

      t_emb = timestep_embedding(tms, self.model_channels).cast(dtypes.float16)
      emb = t_emb.sequential(self.time_embed)

      assert y.shape[0] == x.shape[0]
      emb = emb + y.sequential(self.label_emb[0])

      def run(x:Tensor, bb) -> Tensor:
         if isinstance(bb, UNet.ResBlock): x = bb(x, emb)
         elif isinstance(bb, UNet.SpatialTransformer): x = bb(x, ctx)
         else: x = bb(x)
         return x

      # saved_inputs = []
      # for b in self.input_blocks:
      #    for bb in b:
      #       x = run(x, bb)
      #    saved_inputs.append(x)
      # for bb in self.middle_block:
      #    x = run(x, bb)

      # saved_inputs = []
      # root = "/home/tobi/repos/tinygrad-ports/weights/pre_out_h"
      # x = Tensor(np.load(f"{root}/h.npy")).realize()
      # for i in range(1024):
      #    filepath = f"{root}/{i}.npy"
      #    if os.path.exists(filepath):
      #       saved_inputs.append(Tensor(np.load(filepath)).realize())
      #    else:
      #       break

      # for b in self.output_blocks:
      #    x = x.cat(saved_inputs.pop(), dim=1)
      #    for bb in b:
      #       x = run(x, bb)

      # x = Tensor(np.load("/home/tobi/repos/tinygrad-ports/weights/last_h.npy"))

      return Tensor(np.load("/home/tobi/repos/tinygrad-ports/weights/post_seq_h.npy"))
      return x.sequential(self.out)


class DiffusionModel:
   def __init__(self, *args, **kwargs):
      self.diffusion_model = UNetModel(*args, **kwargs)
   
   def __call__(self, *args, **kwargs) -> Tensor:
      return self.diffusion_model(*args, **kwargs)


class Embedder(ABC):
   input_key: str
   @abstractmethod
   def __call__(self, x:Tensor) -> Tensor:
      pass


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

      def __call__(self, input_ids:Tensor) -> Tensor:
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
   
   def __call__(self, text:Tensor) -> Tensor:
      tokens = self.tokenizer.encode(text)
      return self.transformer.text_model(Tensor(tokens).reshape(1,-1))


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
         q_b, k_b, v_b = self.in_proj_bias  .chunk(3, dim=0)
         q_w, k_w, v_w = self.in_proj_weight.chunk(3, dim=0)
         q,k,v = x.linear(q_w.transpose(), q_b), x.linear(k_w.transpose(), k_b), x.linear(v_w.transpose(), v_b)
         q,k,v = [y.reshape(x.shape[0], -1, self.n_heads, self.d_head).transpose(0, 2) for y in (q,k,v)]
         attn_output = Tensor.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
         return self.out_proj(attn_output.transpose(0, 2).reshape(x.shape))


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
      def __init__(self, dims:int, vocab_size:int=49408, n_heads:int=16, ctx_length:int=77, layers:int=32):
         self.token_embedding = Embedding(vocab_size, dims)
         self.positional_embedding = Tensor.empty(ctx_length, dims)
         self.transformer = Open.ClipTransformer(dims, layers, n_heads)
         self.ln_final = LayerNorm(dims)
         self.text_projection = Tensor.empty(dims, dims)
      
      @property
      def attn_mask(self) -> Tensor:
         if not hasattr(self, "_attn_mask"):
            self._attn_mask = Tensor.full((77, 77), float("-inf")).triu(1)
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
   # layer = "penultimate"
   # always_return_pooled = True
   def __init__(self, dims:int=1280):
      self.model = Open.ClipTextTransformer(dims)
      self.input_key = "txt"
      self.tokenizer = Closed.ClipTokenizer()
   
   def text_transformer_forward(self, x:Tensor, attn_mask:Optional[Tensor]=None):
      penultimate = None
      for i, r in enumerate(self.model.transformer.resblocks):
         if i == len(self.model.transformer.resblocks) - 1:
            penultimate = x
         x = r(x, attn_mask=attn_mask)
      assert penultimate is not None, "should have saved a penultimate"
      return x.permute(1,0,2), penultimate.permute(1,0,2)

   def __call__(self, text:Tensor) -> Tensor:
      tokens = Tensor(self.tokenizer.encode(text)).reshape(1,-1)
      x = self.model.token_embedding(tokens).add(self.model.positional_embedding).permute(1,0,2)
      x, penultimate = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
      x = self.model.ln_final(x)
      pooled = x[Tensor.arange(x.shape[0]), tokens.argmax(axis=-1).numpy().item()] @ self.model.text_projection

      return pooled, penultimate


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
      emb = emb.reshape((b,-1))
      return emb


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L71
class Conditioner:
   OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
   KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}
   embedders: List[Embedder]

   def __init__(self, concat_embedders:List[str]):
      self.embedders = [
         FrozenClosedClipEmbedder(),
         FrozenOpenClipEmbedder(),
      ]
      for input_key in concat_embedders:
         self.embedders.append(ConcatTimestepEmbedderND(256, input_key))
   
   def get_keys(self) -> Set[str]:
      return set(e.input_key for e in self.embedders)
   
   def __call__(self, batch:Dict) -> Dict[str,Tensor]:
      output: Dict[str,Tensor] = {}

      for embedder in self.embedders:
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
      def __init__(self, in_dim:Tensor, out_dim:Optional[Tensor]=None):
         out_dim = out_dim if out_dim is not None else in_dim

         self.norm1 = GroupNorm(32, in_dim)
         self.conv1 = Conv2d(in_dim, out_dim, 3, stride=1, padding=1)
         self.norm2 = GroupNorm(32, out_dim)
         self.conv2 = Conv2d(out_dim, out_dim, 3, stride=1, padding=1)
         self.nin_shortcut = tensor_identity if in_dim == out_dim else Conv2d(in_dim, out_dim, 1, stride=1, padding=0)

      def __call__(self, x:Tensor) -> Tensor:
         h = x.sequential([self.norm1, Tensor.swish, self.conv1])
         h = h.sequential([self.norm2, Tensor.swish, self.conv2])
         return self.nin_shortcut(x) + h


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
         self.block_1 = FirstStage.ResnetBlock(block_in)
         self.attn_1  = FirstStage.AttnBlock  (block_in)
         self.block_2 = FirstStage.ResnetBlock(block_in)


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
            for _ in range(num_res_blocks):
               block.append(FirstStage.ResnetBlock(block_in, block_out))
               block_in = block_out
            
            downsample = tensor_identity if (i_level == len(ch_mult)-1) else FirstStage.Downsample(block_in)
            self.down.append(BlockEntry(block, downsample)) # type: ignore
         
         self.mid = FirstStage.MidEntry(block_in)

         self.norm_out = GroupNorm(32, block_in)
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
            def __init__(self, block:List[FirstStage.ResnetBlock], upsample:Callable[[Any],Any]):
               self.block = block
               self.upsample = upsample
         self.up: List[BlockEntry] = []
         for i_level in reversed(range(len(ch_mult))):
            block = []
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
               block.append(FirstStage.ResnetBlock(block_in, block_out))
               block_in = block_out
            
            upsample = tensor_identity if i_level == 0 else FirstStage.Upsample(block_in)
            self.up.insert(0, BlockEntry(block, upsample)) # type: ignore
         
         self.norm_out = GroupNorm(32, block_in)
         self.conv_out = Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
      
      def __call__(self, z:Tensor) -> Tensor:

         h = z.sequential([self.conv_in, self.mid.block_1, self.mid.attn_1, self.mid.block_2])

         for up in self.up[::-1]:
            for block in up.block:
               h = block(h)
            h = up.upsample(h)

         h = h.sequential([self.norm_out, Tensor.swish, self.conv_out])
         return h


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L102
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L437
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L508
class FirstStageModel:
   def __init__(self, embed_dim:int=4, **kwargs):
      self.encoder = FirstStage.Encoder(**kwargs)
      self.decoder = FirstStage.Decoder(**kwargs)
      self.quant_conv = Conv2d(2*kwargs["z_ch"], 2*embed_dim, 1)
      self.post_quant_conv = Conv2d(embed_dim, kwargs["z_ch"], 1)

   def decode(self, z:Tensor) -> Tensor:
      dec = self.post_quant_conv(z)
      dec = self.decoder(dec)
      return dec


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/discretizer.py#L42
class LegacyDDPMDiscretization:
   def __init__(self, linear_start:float=0.00085, linear_end:float=0.0120, num_timesteps:int=1000):
      self.num_timesteps = num_timesteps
      betas = np.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=np.float32) ** 2
      alphas = 1.0 - betas
      self.alphas_cumprod = np.cumprod(alphas, axis=0)

   def __call__(self, n:int) -> Tensor:
      if n < self.num_timesteps:
         timesteps = np.linspace(self.num_timesteps - 1, 0, n, endpoint=False).astype(int)[::-1]
         alphas_cumprod = self.alphas_cumprod[timesteps]
      elif n == self.num_timesteps:
         alphas_cumprod = self.alphas_cumprod
      sigmas = Tensor((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
      sigmas = sigmas.flip((0,)).cat(Tensor.ones((1,)))
      return sigmas


def expand_dims(x:Tensor, t:Tensor) -> Tensor:
   dims_to_append = len(t.shape) - len(x.shape)
   assert dims_to_append >= 0
   return x.reshape(x.shape + (1,)*dims_to_append)


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/denoiser.py#L42
class Denoiser:
   def __init__(self, num_idx:int):
      self.discretization = LegacyDDPMDiscretization()
      self.scaling = Denoiser.eps_scaling
      self.num_idx = num_idx
   
   @staticmethod
   def eps_scaling(sigma:Tensor) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
      c_skip  = (Tensor.ones_like(sigma)).cast(dtypes.float16)
      c_out   = (-sigma).cast(dtypes.float16)
      c_in    = (1 / (sigma**2 + 1.0) ** 0.5).cast(dtypes.float16)
      c_noise = (sigma).cast(dtypes.float16)
      return c_skip, c_out, c_in, c_noise

   _sigmas = None
   @property
   def sigmas(self) -> Tensor:
      if self._sigmas is None:
         self._sigmas = self.discretization(self.num_idx)
      return self._sigmas

   def sigma_to_idx(self, sigma:Tensor) -> Tensor:
      dists = sigma - self.sigmas[:, None]
      return dists.abs().argmin(axis=0).view(*sigma.shape)

   def idx_to_sigma(self, idx:Union[Tensor,int]) -> Tensor:
      return self.sigmas[idx]

   def __call__(self, model, x:Tensor, sigma:Tensor, cond:Dict) -> Tensor:
      sigma = self.idx_to_sigma(self.sigma_to_idx(sigma))
      sigma_shape = sigma.shape
      sigma = expand_dims(sigma, x)
      c_skip, c_out, c_in, c_noise = self.scaling(sigma)
      # print(f"c_skip: {c_skip.numpy()}")
      # print(f"c_out: {c_out.numpy()}")
      # print(f"c_in: {c_in.numpy()}")
      # print(f"c_noise: {c_noise.numpy()}")
      c_noise = self.sigma_to_idx(c_noise.reshape(sigma_shape))

      c_out  = Tensor(np.load("/home/tobi/repos/tinygrad-ports/weights/last_den_c_out.npy"))
      c_skip = Tensor(np.load("/home/tobi/repos/tinygrad-ports/weights/last_den_c_skip.npy"))
      x      = Tensor(np.load("/home/tobi/repos/tinygrad-ports/weights/last_den_input.npy"))
      return model(x*c_in, c_noise, cond)*c_out + x*c_skip


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/diffusion.py#L19
class SDXL:
   scale_factor: float = 0.13025

   def __init__(self, config:Dict):
      self.conditioner = Conditioner(**config["conditioner"])
      self.first_stage_model = FirstStageModel(**config["first_stage_model"])
      self.model = DiffusionModel(**config["model"])
      self.denoiser = Denoiser(**config["denoiser"])

   # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L543
   def decode(self, x:Tensor) -> Tensor:
      return self.first_stage_model.decode(1.0 / self.scale_factor * x)


def fp16(x:Optional[Tensor]) -> Optional[Tensor]:
   if x is None:
      return x
   return x
   return x.cast(dtypes.float16) # .realize()

def real(x:Optional[Tensor]) -> Optional[Tensor]:
   if x is None:
      return x
   return x.realize()



class VanillaCFG:
   def __init__(self, scale:float):
      self.scale = scale

   def prepare_inputs(self, x:Tensor, s:float, c:Dict, uc:Dict) -> Tuple[Tensor,Tensor,Tensor]:
      c_out = {}
      for k in c:
         assert k in ["vector", "crossattn", "concat"]
         c_out[k] = fp16(Tensor.cat(uc[k], c[k], dim=0))
      return fp16(Tensor.cat(x, x)), fp16(Tensor.cat(s, s)), c_out

   def __call__(self, x:Tensor, sigma:float) -> Tensor:
      x_u, x_c = x.chunk(2)
      x_pred = x_u + self.scale*(x_c - x_u)
      return x_pred


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/sampling.py#L21
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/sampling.py#L287
class DPMPP2MSampler:
   def __init__(self, cfg_scale:float):
      self.discretization = LegacyDDPMDiscretization()
      self.guider = VanillaCFG(cfg_scale)

   def sampler_step(self, old_denoised:Optional[Tensor], prev_sigma:Optional[Tensor], sigma:Tensor, next_sigma:Tensor, denoiser, x:Tensor, c:Dict, uc:Dict) -> Tuple[Tensor,Tensor]:
      old_denoised = fp16(old_denoised)
      prev_sigma   = fp16(prev_sigma)
      sigma        = fp16(sigma)
      next_sigma   = fp16(next_sigma)
      x            = fp16(x)

      denoised = denoiser(*self.guider.prepare_inputs(x, sigma, c, uc))
      # denoised = Tensor(np.load("/home/tobi/repos/tinygrad-ports/weights/pre_guider.npy"))
      denoised = self.guider(denoised, sigma)

      t, t_next = sigma.log().neg(), next_sigma.log().neg()
      h = t_next - t
      r = None if prev_sigma is None else (t - prev_sigma.log().neg()) / h

      mults = [t_next.neg().exp()/t.neg().exp(), (-h).exp().sub(1)]
      if r is not None:
         mults.extend([1 + 1/(2*r), 1/(2*r)])
      mults = [expand_dims(m, x) for m in mults]

      x_standard = mults[0]*x - mults[1]*denoised
      if (old_denoised is None) or (next_sigma.sum().numpy().item() < 1e-14):
         return x_standard, denoised
      
      denoised_d = mults[2]*denoised - mults[3]*old_denoised
      x_advanced = mults[0]*x        - mults[1]*denoised_d
      x = Tensor.where(expand_dims(next_sigma, x) > 0.0, x_advanced, x_standard)
      return x, denoised

   def __call__(self, denoiser, x:Tensor, c:Dict, uc:Dict, num_steps:int) -> Tensor:
      sigmas = self.discretization(num_steps)
      x *= Tensor.sqrt(1.0 + sigmas[0] ** 2.0)
      num_sigmas = len(sigmas)
      s_in = Tensor.ones([x.shape[0]])

      for v in c .values(): real(v)
      for v in uc.values(): real(v)

      # @TinyJit
      # def run(sampler, *args, **kwargs):
      #    sigma, next_sigma = kwargs.pop("sigmas").chunk(2)
      #    out = sampler(
      #       *args,
      #       c=c,
      #       uc=uc,
      #       prev_sigma=real(None if i==0 else s_in*sigmas[i-1]),
      #       sigma=sigma,
      #       next_sigma=next_sigma,
      #       denoiser=denoiser,
      #       **kwargs
      #    )
      #    for e in out:
      #       e.realize()
      #    return out

      # old_denoised = None
      # for i in trange(num_sigmas - 1):
      #    x, old_denoised = run(
      #       self.sampler_step,
      #       old_denoised=real(old_denoised),
      #       sigmas=real((s_in*sigmas[i]).cat(s_in*sigmas[i+1])),
      #       x=real(x),
      #    )

      root = "/home/tobi/repos/tinygrad-ports/weights/last_stage"
      kwargs = {}
      for f in os.listdir(root):
         comps = f.replace(".npy","").split("__")
         data = Tensor(np.load(f"{root}/{f}"))
         if len(comps) == 1:
            kwargs[comps[0]] = data
         elif len(comps) == 2:
            inner = kwargs.get(comps[0], {})
            inner[comps[1]] = data
            kwargs[comps[0]] = inner

      x, _ = self.sampler_step(**kwargs, denoiser=denoiser)

      return x


if __name__ == "__main__":
   Tensor.no_grad = True

   weight_path = os.path.join(os.path.dirname(__file__), "..", "weights", "sd_xl_base_1.0.safetensors")
   state_dict = safe_load(weight_path)

   model = SDXL(configs["SDXL_Base"])

   # model_keys = set(get_state_dict(model).keys())
   # weight_keys = set(state_dict.keys())
   # print(f"model_keys:  {len(model_keys)}")
   # print(f"weight_keys: {len(weight_keys)}")
   # print(f"intersect:   {len(model_keys.intersection(weight_keys))}")
   # print("\nModel Only Keys:")
   # for k in sorted(model_keys.difference(weight_keys)):
   #    print(k)
   # print("\nWeight Only Keys:")
   # for k in sorted(weight_keys.difference(model_keys)):
   #    print(k)
   # assert False

   load_state_dict(model, state_dict, strict=True, apply_fnx=(lambda x: x.cast(dtypes.float16)))
   print("loaded state dict")

   # sampling params
   # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/api.py#L52
   pos_prompt = "a horse sized cat eating a bagel"
   neg_prompt = ""
   img_width  = 1024
   img_height = 1024
   steps = 20
   cfg_scale = 6.0
   eta = 1.0
   aesthetic_score = 5.0

   N = 1
   C = 4
   F = 8

   # # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/helpers.py#L173
   # batch_c : Dict = {
   #    "txt": pos_prompt,
   #    "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
   #    "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
   #    "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
   #    "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
   # }
   # batch_uc: Dict = {
   #    "txt": neg_prompt,
   #    "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
   #    "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
   #    "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
   #    "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
   # }
   # print("starting batch creation")
   # c, uc = model.conditioner(batch_c), model.conditioner(batch_uc)
   # for v in c .values(): v.realize()
   # for v in uc.values(): v.realize()
   # print("created batches")

   # del model.conditioner


   # # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/helpers.py#L101
   # shape = (N, C, img_height // F, img_width // F)
   # randn = Tensor.randn(shape)

   c, uc = {}, {} # type: ignore
   root = "/home/tobi/repos/tinygrad-ports/weights/inputs"
   randn = Tensor(np.load(f"{root}/randn.npy"))
   for f in os.listdir(root):
      if f.startswith("c_"):  c [f.replace("c_", "").split(".")[0]] = Tensor(np.load(f"{root}/{f}"))
      if f.startswith("uc_"): uc[f.replace("uc_","").split(".")[0]] = Tensor(np.load(f"{root}/{f}"))

   def denoiser(x:Tensor, sigma:Tensor, c:Dict) -> Tensor:
      return model.denoiser(model.model, x, sigma, c)

   sampler = DPMPP2MSampler(cfg_scale)
   z = sampler(denoiser, randn, c, uc, steps)

   # z = Tensor(np.load("/home/tobi/repos/tinygrad-ports/weights/samples_z.npy"))

   print("created samples")
   x = model.decode(z)
   print("decoded samples")

   # make image correct size and scale
   x = (x + 1.0) / 2.0
   x = x.reshape(3,img_height,img_width).permute(1,2,0).clip(0,1)*255
   x = x.cast(dtypes.float32).realize().cast(dtypes.uint8)
   print(x.shape)
   
   im = Image.fromarray(x.numpy().astype(np.uint8, copy=False))
   im.show()
