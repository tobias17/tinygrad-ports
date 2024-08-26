from tinygrad import Tensor # type: ignore
from extra.models.clip import OpenClipEncoder, Tokenizer, clip_configs # type: ignore

from PIL import Image

SPECIFIC_I = 1
MAX_INPUTS = 2

def main():
  clip_enc = OpenClipEncoder(**clip_configs["ViT-H-14"]).load_from_pretrained()
  tokenizer = Tokenizer.ClipTokenizer()

  with open("inputs/captions.txt", "r") as f:
    texts = f.read().split("\n")

  tokens, images = [], []
  for i, text in enumerate(texts):
    if SPECIFIC_I >= 0 and i != SPECIFIC_I:
      continue
    tokens.append( Tensor(tokenizer.encode(text, pad_with_zeros=True))      .unsqueeze(0) )
    images.append( clip_enc.prepare_image(Image.open(f"inputs/gen_{i}.png")).unsqueeze(0) )
    if MAX_INPUTS > 0 and len(tokens) >= MAX_INPUTS:
      break

  score = clip_enc.get_clip_score(Tensor.cat(*tokens, dim=0).realize(), Tensor.cat(*images, dim=0).realize())
  print(f"score: {score.numpy()}")

if __name__ == "__main__":
  main()
