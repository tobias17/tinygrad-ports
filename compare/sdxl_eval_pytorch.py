from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import torch
from PIL import Image
import numpy as np

GUIDANCE = 8.0
STEPS = 20

def main():
  torch.manual_seed(42)
  assert torch.cuda.is_available(), "cuda was not available"

  model_id = "stabilityai/stable-diffusion-xl-base-1.0"
  scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
  pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    safety_checker=None,
    add_watermark=False,
    variant="fp16",
    torch_dtype=torch.float16,
    device='cuda',
  )
  pipe.to('cuda')

  pos_prompt = "a horse sized cat eating a bagel"
  neg_prompt = ""

  with torch.no_grad():
    pos_prompt_embeds_list = []
    neg_prompt_embeds_list = []
    for tokenizer, text_encoder in zip([pipe.tokenizer, pipe.tokenizer_2], [pipe.text_encoder, pipe.text_encoder_2]):
      pos_text_input = tokenizer(pos_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
      pos_prompt_embeds = text_encoder(pos_text_input.input_ids.cuda(), output_hidden_states=True)
      pos_prompt_embeds_list.append(pos_prompt_embeds.hidden_states[-2])
      pos_pooled_embeds = pos_prompt_embeds[0]

      neg_text_input = tokenizer(neg_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
      neg_prompt_embeds = text_encoder(neg_text_input.input_ids.cuda(), output_hidden_states=True)
      neg_prompt_embeds_list.append(neg_prompt_embeds.hidden_states[-2])
      neg_pooled_embeds = pos_prompt_embeds[0]

    pos_prompt_embeds = torch.concat(pos_prompt_embeds_list, dim=-1)
    neg_prompt_embeds = torch.concat(neg_prompt_embeds_list, dim=-1)

    print(f"pos_embed_shape: {pos_prompt_embeds.shape}")
    print(f"neg_embed_shape: {neg_prompt_embeds.shape}")
    print(f"pos_pooled_shape: {pos_pooled_embeds.shape}")
    print(f"neg_pooled_shape: {neg_pooled_embeds.shape}")

    torch.manual_seed(42)
    noise = torch.randn(1, 4, 128, 128).half().cuda()
    np.save("out_noise.npy", noise.cpu().numpy())

    generated: torch.Tensor = pipe(
      # prompt=pos_prompt,
      # prompt_2=pos_prompt,
      # negative_prompt=neg_prompt,
      # negative_prompt_2=neg_prompt,
      prompt_embeds=pos_prompt_embeds,
      negative_prompt_embeds=neg_prompt_embeds,
      pooled_prompt_embeds=pos_pooled_embeds,
      negative_pooled_prompt_embeds=neg_pooled_embeds,
      guidance_scale=GUIDANCE,
      num_inference_steps=STEPS,
      output_type="pt",
      latents=noise,
    ).images

    img = Image.fromarray(generated.squeeze(0).permute(1, 2, 0).mul(255.0).to(torch.uint8).detach().cpu().numpy())
    img.save("out_img_pytorch.png")

if __name__ == "__main__":
  main()
