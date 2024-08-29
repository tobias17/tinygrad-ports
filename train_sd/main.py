from tinygrad import Tensor, dtypes, TinyJit, Device # type: ignore
from tinygrad.nn.optim import AdamW, Adam, SGD # type: ignore
from tinygrad.helpers import getenv, BEAM # type: ignore
# from tinygrad.helpers import tqdm
from tqdm import tqdm # type: ignore
import matplotlib.pyplot as plt

import time, os, datetime, math
from typing import Dict, Tuple
from PIL import Image
import pandas as pd # type: ignore
import numpy as np

from tinygrad.nn.state import load_state_dict, torch_load, get_parameters, get_state_dict, safe_save, safe_load # type: ignore

from unet import UNetModel, timestep_embedding # type: ignore
from clip import OpenClipEncoder, clip_configs, Tokenizer # type: ignore
from inception import FidInceptionV3 # type: ignore
from examples.sdv2 import params, get_alphas_cumprod # type: ignore
from ddim import DdimSampler
# from inception import InceptionV3

# TODO:
# - Figure out AdamW
# - Investigate Async Beam
# - Inhouse this god-awful webdataset module



from examples.sdv2 import DiffusionModel, AutoencoderKL, FrozenOpenClipEmbedder, LegacyDDPMDiscretization, append_dims, run
class StableDiffusionV2:
  def __init__(self, unet_config:Dict, cond_stage_config:Dict, parameterization:str="v"):
    self.model             = DiffusionModel(UNetModel(**unet_config))
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model  = FrozenOpenClipEmbedder(**cond_stage_config)
    self.alphas_cumprod    = get_alphas_cumprod()
    self.parameterization  = parameterization

    self.discretization = LegacyDDPMDiscretization()
    self.sigmas = self.discretization(1000, flip=True)

  def denoise(self, x:Tensor, sigma:Tensor, cond:Dict) -> Tensor:

    def sigma_to_idx(s:Tensor) -> Tensor:
      dists = s - self.sigmas.unsqueeze(1)
      return dists.abs().argmin(axis=0).view(*s.shape)

    sigma = self.sigmas[sigma_to_idx(sigma)]
    sigma_shape = sigma.shape
    sigma = append_dims(sigma, x)

    c_skip = 1.0 / (sigma**2 + 1.0)
    c_out = -sigma / (sigma**2 + 1.0) ** 0.5
    c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
    c_noise = sigma_to_idx(sigma.reshape(sigma_shape))

    def prep(*tensors:Tensor):
      return tuple(t.cast(TRAIN_DTYPE).realize() for t in tensors)

    return run(self.model.diffusion_model, *prep(x*c_in, c_noise, cond["crossattn"], c_out, x*c_skip))

  def decode(self, x:Tensor, height:int, width:int) -> Tensor:
    x = self.first_stage_model.post_quant_conv(1/0.18215 * x)
    x = self.first_stage_model.decoder(x)

    # make image correct size and scale
    x = (x + 1.0) / 2.0
    x = x.reshape(3,height,width).permute(1,2,0).clip(0,1).mul(255).cast(dtypes.uint8)
    return x



if __name__ == "__main__":
  seed = 42
  Tensor.manual_seed(seed)

  __OUTPUT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
  def get_output_root(*folders:str) -> str:
    dirpath = os.path.join(__OUTPUT_ROOT, *folders)
    if not os.path.exists(dirpath):
      os.makedirs(dirpath)
    return dirpath

  TRAIN_DTYPE = dtypes.float32

  # GPUS = tuple([f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 1))])

  # GPUS = tuple([f'{Device.DEFAULT}:{i}' for i in [1,2,3,4,5]])
  # DEVICE_BS = 2

  GPUS = tuple([f'{Device.DEFAULT}:{i}' for i in [4,5]])
  DEVICE_BS = 1

  GLOBAL_BS = DEVICE_BS * len(GPUS)
  EVAL_EVERY = math.ceil(512000.0 / GLOBAL_BS)
  print(f"Configured to Eval every {EVAL_EVERY} steps")

  EVAL_DEVICE_BS = 2
  EVAL_GLOBAL_BS = EVAL_DEVICE_BS * len(GPUS)

  # Model, Conditioner, Other Variables

  wrapper_model = StableDiffusionV2(**params)
  # del wrapper_model.model
  # load_state_dict(wrapper_model, torch_load("/home/tiny/tinygrad/weights/512-base-ema.ckpt")["state_dict"], strict=False)
  load_state_dict(wrapper_model, torch_load("/home/tiny/tinygrad/weights/768-v-ema.ckpt")["state_dict"], strict=False)
  for k,w in get_state_dict(wrapper_model).items():
    w_ = w.cast(TRAIN_DTYPE)
    if k.startswith("first_stage_model.") or k.startswith("model."):
      w_ = w_.shard(GPUS, axis=None)
    w.replace(w_).realize()

  model = UNetModel(**params["unet_config"])
  # load_state_dict(model, safe_load("archive/2024-08-18_21-49-34/weights/unet_step057344.safe"), strict=True)
  for w in get_state_dict(model).values():
    w.replace(w.cast(TRAIN_DTYPE).shard(GPUS, axis=None)).realize()
  # optimizer = AdamW(get_parameters(model), lr=1.25e-7, b1=0.9, b2=0.999, weight_decay=0.01)
  # optimizer = AdamW(get_parameters(model), lr=1.25e-7, eps=1.0)
  # optimizer = Adam(get_parameters(model), lr=1.25e-7 *50)
  optimizer = SGD(get_parameters(model), lr=1.25e-7 *10)

  alphas_cumprod      = get_alphas_cumprod()
  alphas_cumprod_prev = Tensor([1.0]).cat(alphas_cumprod[:-1])
  sqrt_alphas_cumprod = alphas_cumprod.sqrt()
  sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()


  # Dataset

  def collate_fnx(batch):
    arr = np.zeros((len(batch),8,64,64), dtype=np.float32)
    for i, e in enumerate(batch):
      arr[i,:,:,:] = e["npy"]
    return { "moments": Tensor(arr), "txt": [e["txt"] for e in batch] }

  def sample_moments(moments:Tensor) -> Tensor:
    mean, logvar = moments.chunk(2, dim=1)
    logvar = logvar.clip(-30.0, 20.0)
    std = Tensor.exp(logvar * 0.5)
    return mean + std * Tensor.rand(*mean.shape)

  from webdataset import WebDataset, WebLoader # type: ignore
  urls = "/home/tiny/tinygrad/datasets/laion-400m/webdataset-moments-filtered/{00000..00831}.tar"
  keep_only_keys = { "npy", "txt" }
  def filter_fnx(sample): return {k:v for k,v in sample.items() if k in keep_only_keys}
  dataset = WebDataset(urls=urls, resampled=True, cache_size=-1, cache_dir=None)
  dataset = dataset.shuffle(size=1000).decode().map(filter_fnx).batched(GLOBAL_BS, partial=False, collation_fn=collate_fnx)
  dataloader = WebLoader(dataset, batch_size=None, shuffle=False, num_workers=1, persistent_workers=True)


  MAX_QUEUE_SIZE = 10

  MAX_ITERS   = 512000
  MEAN_EVERY  = 32
  GRAPH_EVERY = 256
  SAVE_EVERY  = 4096
  ONLY_LAST   = False
  losses = [] # type: ignore
  saved_losses = [] # type: ignore

  BEAM_VAL = BEAM.value
  BEAM.value = 0



  @TinyJit
  def tokenize_step(tokens:Tensor) -> Tensor:
    return wrapper_model.cond_stage_model.embed_tokens(tokens).realize()


  if False:

    @TinyJit
    @Tensor.train()
    def train_step(x:Tensor, x_noisy:Tensor, t_emb:Tensor, c:Tensor) -> Tensor:
      output = model.pre_embedded(x_noisy, t_emb, c)

      loss = (x - output).square().mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      return loss.realize()

    def prep_for_jit(*inputs:Tensor) -> Tuple[Tensor,...]:
      return tuple(i.cast(TRAIN_DTYPE).shard(GPUS, axis=0).realize() for i in inputs)
  
    # Main Train Loop
    for i, entry in enumerate(dataloader):
      if i >= MAX_ITERS:
        break

      st = time.perf_counter()

      c = tokenize_step(Tensor.cat(*[wrapper_model.cond_stage_model.tokenize(t) for t in entry["txt"]]))
      x = (sample_moments(entry["moments"]) * 0.18215)
      t = Tensor.randint(x.shape[0], low=0, high=1000)
      noise = Tensor.randn(x.shape)
      x_noisy =   sqrt_alphas_cumprod          [t].reshape(GLOBAL_BS, 1, 1, 1) * x \
                + sqrt_one_minus_alphas_cumprod[t].reshape(GLOBAL_BS, 1, 1, 1) * noise
      t_emb = timestep_embedding(t, 320).cast(TRAIN_DTYPE)
      inputs = prep_for_jit(x, x_noisy, t_emb, c)

      pt = time.perf_counter()

      BEAM.value = BEAM_VAL
      loss = train_step(*inputs).numpy().item()
      BEAM.value = 0
      losses.append(loss)

      et = time.perf_counter()
      tqdm.write(f"{i:05d}: {(et-st)*1000.0:6.0f} ms run, {(pt-st)*1000.0:6.0f} ms prep, {(et-pt)*1000.0:6.0f} ms step, {loss:>2.5f} train loss")

      if i > 0 and i % MEAN_EVERY == 0:
        saved_losses.append(sum(losses) / len(losses))
        losses = []
      if i > 0 and i % GRAPH_EVERY == 0:
        plt.clf()
        plt.plot(np.arange(1,len(saved_losses)+1)*MEAN_EVERY, saved_losses)
        plt.ylim((0,None))
        plt.xlabel("step")
        plt.ylabel("loss")
        figure = plt.gcf()
        figure.set_size_inches(18/1.5, 10/1.5)
        plt.savefig(os.path.join(get_output_root(), "loss"), dpi=100)
      if i > 0 and i % SAVE_EVERY == 0:
        safe_save(get_state_dict(model), os.path.join(get_output_root("weights"), "unet_last.safe" if ONLY_LAST else f"unet_step{i:06d}.safe"))
      
      if i > 0 and i % EVAL_EVERY == 0:
        pass

  else:

    # Evaluation Loop
    sampler = DdimSampler()
    inception = FidInceptionV3().load_from_pretrained()
    for w in get_state_dict(inception).values():
      w.replace(w.cast(dtypes.float16).shard(GPUS, axis=None)).realize()
    clip_enc  = OpenClipEncoder(**clip_configs["ViT-H-14"]).load_from_pretrained()
    tokenizer = Tokenizer.ClipTokenizer()

    i = 0
    df = pd.read_csv("/home/tiny/tinygrad/datasets/coco2014/val2014_30k.tsv", sep='\t', header=0)
    captions = df["caption"].array
    with open(get_output_root()+"/captions.txt", "w") as f:
      f.write("\n".join(f"{i:05d}: {c}" for i,c in enumerate(captions)))
    inception_activations = []

    assert len(df) % EVAL_GLOBAL_BS == 0, f"eval dataset size ({len(df)}) must be divisible by the EVAL_GLOBAL_BS ({EVAL_GLOBAL_BS})"

    @TinyJit
    def run_inc_and_clip(x:Tensor, tokens:Tensor, images:Tensor) -> Tuple[Tensor,Tensor]:
      return inception(x).realize(), clip_enc.get_clip_score(tokens, images).realize()

    wall_time_start = time.time()

    Tensor.no_grad = True
    while i < len(df):
      texts = captions[i:i+EVAL_GLOBAL_BS]
      texts[0] = "a horse sized cat eating a bagel"
      tokens = [wrapper_model.cond_stage_model.tokenize(t) for t in texts]
      empty_token = wrapper_model.cond_stage_model.tokenize("")

      c  = tokenize_step(Tensor.cat(*tokens).realize()) #.shard(GPUS, axis=0)
      uc = tokenize_step(Tensor.cat(*([empty_token]*EVAL_GLOBAL_BS)).realize()) #.shard(GPUS, axis=0)

      z = sampler.sample(
        wrapper_model.model.diffusion_model, EVAL_GLOBAL_BS, c, uc, num_steps=50,
        shard_fnx=(lambda x: x.shard(GPUS, axis=0)),
        all_fnx_=(lambda x: x.shard_(GPUS, axis=None)),
        dtype=TRAIN_DTYPE,
      )

      x = wrapper_model.first_stage_model.post_quant_conv(1/0.18215 * z)
      x = wrapper_model.first_stage_model.decoder(x)
      x = (x + 1.0) / 2.0
      x.realize()

      # inception_activations.append(inception(x).squeeze(3).squeeze(2).to(Device.DEFAULT).realize())
      # if len(inception_activations) > 20:
      #   inception_activations = [Tensor.cat(*inception_activations, dim=0).realize()]

      images = []
      x_clip = x.to(Device.DEFAULT).realize()
      for j in range(EVAL_GLOBAL_BS):
        im = Image.fromarray(x_clip[j].clip(0,1).mul(255).cast(dtypes.uint8).permute(1,2,0).numpy())
        im.save(get_output_root("images")+f"/{i+j:05d}.png")
        images.append(clip_enc.prepare_image(im).unsqueeze(0))
      # clip_score = clip_enc.get_clip_score(Tensor.cat(*tokens, dim=0).realize(), Tensor.cat(*images, dim=0).realize())
      # print(f"clip_score: {clip_score.mean().numpy():.5f}")

      inc_out, clip_out = run_inc_and_clip(x.realize(), Tensor.cat(*tokens, dim=0).realize(), Tensor.cat(*images, dim=0).realize())

      inception_activations.append(inc_out.squeeze(3).squeeze(2).to(Device.DEFAULT).realize())
      if len(inception_activations) > 20:
        inception_activations = [Tensor.cat(*inception_activations, dim=0).realize()]

      print(f"clip_score: {clip_out.mean().numpy():.5f}")

      i += EVAL_GLOBAL_BS

      def smart_print(sec:float) -> str:
        if sec < 60: return f"{sec:.0f} sec"
        mins = sec / 60.0
        if mins < 60: return f"{mins:.1f} mins"
        hours = mins / 60.0
        return f"{hours:.1f} hours"
      wall_time_delta = time.time() - wall_time_start
      eta_time = (wall_time_delta / (i / len(df))) - wall_time_delta
      print(f"{100.0*i/len(df):02.2f}% ({i}/{len(df)}), elapsed wall time: {smart_print(wall_time_delta)}, eta: {smart_print(eta_time)}")

      input("next? ")

    inception_act = Tensor.cat(*inception_activations, dim=0)
    fid_score = inception.compute_score(inception_act)
    print(f"fid_score:  {fid_score}")
    Tensor.no_grad = False
