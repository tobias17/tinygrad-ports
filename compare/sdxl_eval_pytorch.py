from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import torch

def main():
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

  pos_prompt = "a horse sized cat eating a bagel"
  neg_prompt = ""

  with torch.no_grad():
    pos_prompt_embeds_list = []
    for tokenizer, text_encoder in zip([pipe.tokenizer, pipe.tokenizer_2], [pipe.text_encoder, pipe.text_encoder_2]):
      pos_text_input = tokenizer(pos_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
      text_input_ids = pos_text_input.input_ids
      pos_prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
      pos_pooled_prompt_embeds = pos_prompt_embeds[0]
      pos_prompt_embeds_list.append(pos_prompt_embeds)

      neg_text_input = tokenizer(neg_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")

    pos_prompt_embeds = torch.concat(pos_prompt_embeds_list, dim=-1)

    for 

    prompt_embed = []
    negative_prompt_embed = []
    pooled_prompt_embed = []
    negative_pooled_prompt_embed = []

    generated = pipe(
      prompt_embeds=pos_prompt_embeds,
      negative_prompt_embeds=[negative_prompt_embed],
      pooled_prompt_embeds=[pooled_prompt_embed],
      negative_pooled_prompt_embeds=[negative_pooled_prompt_embed],
      guidance_scale=self.guidance,
      num_inference_steps=self.steps,
      output_type="pt",
      latents=latents_input,
    ).images

if __name__ == "__main__":
  main()
