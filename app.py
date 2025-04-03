import os
import torch
import numpy as np
from PIL import Image
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora
from torchvision import transforms

# Initialize the image processor
base_path = "black-forest-labs/FLUX.1-dev"    
lora_base_path = "./models"

# Ensure the model uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and move to GPU
pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.float16).to(device)  # Use fp16 for reduced memory consumption
transformer = FluxTransformer2DModel.from_pretrained(base_path, subfolder="transformer", torch_dtype=torch.float16).to(device)
pipe.transformer = transformer

# Image transformation to tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor in [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] for many models
])

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

def single_condition_generate_image(prompt, spatial_img_path, height, width, seed, control_type, output_image_path):
    # Load and process the spatial image
    spatial_img = Image.open(spatial_img_path).convert("RGB")
    spatial_img = transform(spatial_img).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension

    # Set the control type (e.g., Ghibli)
    if control_type == "Ghibli":
        lora_path = os.path.join(lora_base_path, "Ghibli.safetensors")
    set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=512)

    # Prepare the generator for the specified seed and move it to the GPU
    generator = torch.Generator(device=device).manual_seed(seed)

    # Process the image
    spatial_imgs = [spatial_img] if spatial_img else []
    image = pipe(
        prompt,
        height=int(height),
        width=int(width),
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=generator,  # Generator now uses the correct device (GPU)
        subject_images=[],
        spatial_images=spatial_imgs,
        cond_size=512,
    ).images[0]

    # Clear the cache after generation
    clear_cache(pipe.transformer)

    # Check if image is generated and save it
    if image:
        print(f"Saving image to: {output_image_path}")
        image.save(output_image_path)
        print(f"Generated image saved to {output_image_path}")
    else:
        print("Image generation failed!")
    
    return image

# Example usage
prompt = "Ghibli Studio style, Charming hand-drawn anime-style illustration"
spatial_img_path = "./test_imgs/00.png"  # Path to the input image
height = 768
width = 768
seed = 42
control_type = "Ghibli"
output_image_path = "/root/flx-gbl/output_image.png"  # Absolute path to the generated image

# Call the function to generate the image and save it
print("Starting image generation...")
single_condition_generate_image(prompt, spatial_img_path, height, width, seed, control_type, output_image_path)
print("Script finished.")
