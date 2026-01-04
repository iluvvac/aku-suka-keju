import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
)

pipe.to("mps") # Ganti "mps" menjadi "cuda" jika menggunakan NVIDIA

init_image = Image.open("product.jpg").convert("RGB").resize((512, 512))
prompt = "cinematic shot of a shiny luxurious product, studio lighting, 8k"

print("Generating...")
with torch.inference_mode():
    image = pipe(
        prompt, 
        image=init_image, 
        num_inference_steps=2,  
        strength=0.5,           
        guidance_scale=0.0      
    ).images[0]

image.save("hasil_turbo.png")
print("Selesai! Cek hasil_turbo.png")