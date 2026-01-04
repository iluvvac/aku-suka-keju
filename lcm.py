import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

model_id = "Lykon/dreamshaper-7"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    safety_checker=None 
)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora() 

pipe.to("mps") # Ganti "mps" menjadi "cuda" jika menggunakan NVIDIA
pipe.enable_attention_slicing() 

init_image = Image.open("product.jpg").convert("RGB").resize((512, 512))
prompt = "shiny luxurious product, professional photography, studio lighting"

print("Generating mode ringan...")
with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        image=init_image,
        num_inference_steps=4,  
        strength=0.6,
        guidance_scale=1.0      
    ).images[0]

image.save("hasil_lcm_ringan.png")
print("Selesai! Cek hasil_lcm_ringan.png")