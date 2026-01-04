import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

model_id = "timbrooks/instruct-pix2pix"
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
)

device = "mps" # Ganti "mps" menjadi "cuda" jika menggunakan NVIDIA
pipeline.to(device)
pipeline.enable_attention_slicing() 

image = Image.open("product.jpg").convert("RGB")
prompt = "Make the product look shiny and luxurious, cinematic lighting"

print("Processing...")
with torch.inference_mode():
    output = pipeline(
        prompt=prompt,
        image=image,
        num_inference_steps=20, 
        image_guidance_scale=1.2, 
        guidance_scale=7.5
    ).images[0]

output.save("output_lightweight.png")
print("Selesai! Disimpan di output_lightweight.png")