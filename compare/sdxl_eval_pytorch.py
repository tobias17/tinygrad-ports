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
  )

  with torch.no_grad():
    

    prompt_embed = []
    negative_prompt_embed = []
    pooled_prompt_embed = []
    negative_pooled_prompt_embed = []

    generated = self.pipe(
      prompt_embeds=[prompt_embed],
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
