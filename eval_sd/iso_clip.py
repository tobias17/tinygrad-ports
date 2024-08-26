from tinygrad import Tensor # type: ignore
from extra.models.clip import OpenClipEncoder, Tokenizer, clip_configs # type: ignore

from PIL import Image

def main():
  clip_enc = OpenClipEncoder(**clip_configs["ViT-H-14"]).load_from_pretrained()
  tokenizer = Tokenizer.ClipTokenizer()

  tokens = Tensor(tokenizer.encode("A city at night with people walking around.", pad_with_zeros=True))
  images = clip_enc.prepare_image(Image.open("inputs/gen_0.png"))

  score = clip_enc.get_clip_score(tokens.unsqueeze(0).realize(), images.unsqueeze(0).realize())
  print(f"score: {score.numpy()}")

if __name__ == "__main__":
  main()
