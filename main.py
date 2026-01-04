import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511", 
    torch_dtype=torch.float16
)
print("pipeline loaded")

try:
    pipeline.enable_model_cpu_offload() 
except:
    pipeline.to("mps") # Ganti "mps" menjadi "cuda" jika menggunakan NVIDIA

image1 = Image.open("product.jpg")
prompt = "Make the product look more shiny and luxurious, photoshoot in studio."

inputs = {
    "image": [image1], 
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": "ugly, blurry, low quality",
    "num_inference_steps": 30, 
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

print("Starting inference...")
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_product_edit.png")
    print("Success! Image saved at", os.path.abspath("output_product_edit.png"))